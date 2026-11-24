"""Lab 03: MLP vs CNN vs attention on Fashion-MNIST, and what each one costs.

Produces in lab03/results/:
  model_summaries.txt    torchinfo.summary of each model
  flops_per_layer.csv    hand-calculated vs thop FLOPs for one layer of each type
  flops_per_model.csv    hand-calculated vs thop vs torchinfo FLOPs for whole models
  latency.csv            inference latency, batch 1 (median of 100) and 128 (median of 20)
  training_log.csv       per-epoch train loss, val accuracy, epoch time
  results.csv            params, FLOPs, latency, test accuracy per model
  trade_offs.png         test accuracy vs parameters and vs latency
"""
import os
import platform
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from thop import profile
from torch.utils.data import DataLoader, random_split
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
LATENCY_RUNS = {1: 100, 128: 20}
WARMUP_RUNS = 10
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = Path(__file__).resolve().parents[1] / "results"
COLORS = {"MLP": "#2a78d6", "CNN": "#eb6834", "Attention": "#1baf7a"}
MARKERS = {"MLP": "o", "CNN": "s", "Attention": "^"}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# --------------------------------------------------------------- data (task 1)
def load_data():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    full_train = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    train, val = random_split(full_train, [55_000, 5_000],
                              generator=torch.Generator().manual_seed(SEED))
    loader_gen = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=loader_gen),
        DataLoader(val, batch_size=BATCH_SIZE),
        DataLoader(test, batch_size=BATCH_SIZE),
    )


# ------------------------------------------------------------- models (task 2)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    """28x28 -> 16 patches of 7x7 -> Linear(49, 64) -> 2 encoder layers -> mean pool -> 10."""

    def __init__(self, patch=7, d_model=64, nhead=4, dim_ff=128, n_layers=2):
        super().__init__()
        self.patch = patch
        self.seq = (28 // patch) ** 2
        self.proj = nn.Linear(patch * patch, d_model)
        # Not in the manual: without a position signal the encoder is permutation-
        # invariant and cannot tell where a patch came from. Adds 16*64 = 1,024 params.
        self.pos = nn.Parameter(torch.zeros(1, self.seq, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff,
                                           batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def patchify(self, x):
        b, p = x.shape[0], self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, 1, 4, 4, 7, 7)
        return x.reshape(b, self.seq, p * p)  # row-major patch order

    def forward(self, x):
        z = self.proj(self.patchify(x)) + self.pos
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def build_models():
    set_seeds()
    return {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}


# ------------------------------------------------------- cost (task 3: FLOPs)
def linear_flops(n_in, n_out, rows=1):
    return 2 * n_in * n_out * rows


def conv_flops(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def encoder_layer_flops(seq, d, ff):
    """Ch. 4 formulas for one Transformer encoder layer (matmul FLOPs only)."""
    return {
        "QKV projections": 2 * seq * d * 3 * d,
        "attention scores QK^T": 2 * seq * seq * d,
        "attention-weighted sum AV": 2 * seq * seq * d,
        "output projection": 2 * seq * d * d,
        "feed-forward": 2 * seq * d * ff + 2 * seq * ff * d,
    }


def hand_flops_per_model():
    enc = sum(encoder_layer_flops(16, 64, 128).values())
    return {
        "MLP": linear_flops(784, 256) + linear_flops(256, 128) + linear_flops(128, 10),
        "CNN": conv_flops(3, 1, 16, 28, 28) + conv_flops(3, 16, 32, 14, 14)
        + linear_flops(32 * 7 * 7, 10),
        "Attention": linear_flops(49, 64, rows=16) + 2 * enc + linear_flops(64, 10),
    }


def thop_flops(module, x):
    """thop reports MACs; x2 to compare with FLOPs."""
    macs, _, layer_info = profile(module, inputs=(x,), verbose=False, ret_layer_info=True)
    return 2 * macs, layer_info


def torch_flops(module, *inputs):
    """FLOPs seen by PyTorch's own dispatcher-level counter.

    Run in train mode: in eval mode nn.TransformerEncoderLayer takes a fused
    "fast path" kernel that the counter (like thop) does not see. Dropout
    doesn't change the FLOP count, and no_grad keeps it to the forward pass.
    """
    was_training = module.training
    module.train()
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        module(*inputs)
    module.train(was_training)
    return counter.get_total_flops()


def flatten_layer_info(info, prefix=""):
    rows = []
    for name, (macs, params, children) in info.items():
        full = f"{prefix}{name}"
        rows.append((full, macs))
        rows += flatten_layer_info(children, full + ".")
    return rows


def compute_costs(models):
    x1 = torch.zeros(1, 1, 28, 28)
    hand = hand_flops_per_model()
    lines, model_rows, layer_rows = [], [], []

    for name, model in models.items():
        model.eval()
        s = summary(model, input_size=(1, 1, 28, 28), verbose=0,
                    col_names=("output_size", "num_params", "mult_adds"))
        lines.append(f"===== {name} =====\n{s}\n")
        thop_total, info = thop_flops(model, x1)
        torch_total = torch_flops(model, x1)
        model.eval()
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            model(x1)
        breakdown = "; ".join(f"{k}={2 * v:,.0f}" for k, v in flatten_layer_info(info) if v)
        model_rows.append({
            "model": name,
            "params_torchinfo": s.total_params,
            "hand_flops": hand[name],
            "thop_flops": thop_total,
            "thop_over_hand": thop_total / hand[name],
            "torch_counter_flops": torch_total,
            "torch_counter_eval_flops": counter.get_total_flops(),
            "torchinfo_flops": 2 * s.total_mult_adds,
            "torchinfo_over_hand": 2 * s.total_mult_adds / hand[name],
            "thop_breakdown": breakdown,
        })

    # One layer of each type, profiled on its own input.
    mlp, cnn, att = models["MLP"], models["CNN"], models["Attention"]
    att_in = torch.zeros(1, 16, 64)
    enc_parts = encoder_layer_flops(16, 64, 128)
    singles = [
        ("Linear  MLP fc1 784->256", mlp.net[1], torch.zeros(1, 784),
         linear_flops(784, 256), "2*784*256"),
        ("Conv2d  CNN conv2 16->32, 3x3, out 14x14", cnn.net[3], torch.zeros(1, 16, 14, 14),
         conv_flops(3, 16, 32, 14, 14), "2*3*3*16*32*14*14"),
        ("Attention  encoder layer 1 (seq 16, d 64, ff 128)", att.encoder.layers[0], att_in,
         sum(enc_parts.values()), " + ".join(f"{k} {v:,}" for k, v in enc_parts.items())),
        ("  - its self-attention block only", att.encoder.layers[0].self_attn, None,
         sum(v for k, v in enc_parts.items() if k != "feed-forward"), "QKV + QK^T + AV + out"),
    ]
    for label, module, x, hand_f, formula in singles:
        inputs = (att_in, att_in, att_in) if x is None else (x,)  # MHA needs (q, k, v)
        macs, _ = profile(module, inputs=inputs, verbose=False)
        thop_f = 2 * macs
        torch_f = torch_flops(module, *inputs)
        layer_rows.append({
            "layer": label, "formula": formula, "hand_flops": hand_f,
            "thop_flops": thop_f, "thop_over_hand": thop_f / hand_f,
            "torch_counter_flops": torch_f, "torch_counter_over_hand": torch_f / hand_f,
        })

    (RESULTS / "model_summaries.txt").write_text("\n".join(lines), encoding="utf-8")
    return pd.DataFrame(model_rows), pd.DataFrame(layer_rows)


# ----------------------------------------------------- cost (task 3: latency)
@torch.inference_mode()
def measure_latency(model, test_loader):
    model.eval()
    images, _ = next(iter(test_loader))
    rows = []
    for batch, runs in LATENCY_RUNS.items():
        x = images[:batch].contiguous()
        for _ in range(WARMUP_RUNS):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)
        med = statistics.median(times) * 1e3
        rows.append({"batch": batch, "runs": runs, "median_ms": med,
                     "min_ms": min(times) * 1e3, "max_ms": max(times) * 1e3,
                     "per_image_ms": med / batch})
    return rows


# ------------------------------------------------------------ training (task 4)
@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total


def train(name, model, train_loader, val_loader):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.perf_counter()
        total_loss = n = 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(y)
            n += len(y)
        epoch_s = time.perf_counter() - t0
        val_acc = evaluate(model, val_loader)
        log.append({"model": name, "epoch": epoch, "train_loss": total_loss / n,
                    "val_acc": val_acc, "epoch_s": epoch_s})
        print(f"  {name:<9} epoch {epoch}  loss={total_loss / n:.4f}  "
              f"val_acc={val_acc:.4f}  {epoch_s:.1f}s")
    return log


# ------------------------------------------------------------- plots (task 5)
def plot_trade_offs(res):
    panels = [
        ("params", "Parameters", "{:,.0f}"),
        ("latency_b1_ms", "Latency, batch 1 (ms, median of 100)", "{:.3f} ms"),
        ("latency_b128_ms", "Latency, batch 128 (ms, median of 20)", "{:.2f} ms"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (col, xlabel, fmt) in zip(axes, panels):
        for _, r in res.iterrows():
            ax.scatter(r[col], r.test_acc * 100, s=90, color=COLORS[r.model],
                       marker=MARKERS[r.model], edgecolor="white", linewidth=2,
                       zorder=3, label=r.model)
            ax.annotate(f"{r.model}\n{fmt.format(r[col])}, {r.test_acc:.2%}",
                        (r[col], r.test_acc * 100), textcoords="offset points",
                        xytext=(8, -4), fontsize=8.5, color="#333333", va="top")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel + " — log scale")
        ax.grid(alpha=0.25, which="both")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        lo, hi = res[col].min(), res[col].max()
        ax.set_xlim(lo / 1.6, hi * 3.5)  # room for the direct labels
    acc = res.test_acc * 100
    axes[0].set_ylim(acc.min() - 1.0, acc.max() + 0.5)  # room under the lowest label
    axes[0].set_ylabel("Test accuracy (%) after 5 epochs")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Fashion-MNIST: accuracy vs model size and inference latency (CPU)")
    fig.tight_layout()
    fig.savefig(RESULTS / "trade_offs.png", dpi=150)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | {platform.processor()} | "
          f"{torch.get_num_threads()} torch threads / {os.cpu_count()} logical cores")
    train_loader, val_loader, test_loader = load_data()
    print(f"train {len(train_loader.dataset)}  val {len(val_loader.dataset)}  "
          f"test {len(test_loader.dataset)}")
    models = build_models()

    model_df, layer_df = compute_costs(models)
    layer_df.to_csv(RESULTS / "flops_per_layer.csv", index=False)
    model_df.to_csv(RESULTS / "flops_per_model.csv", index=False)
    print(layer_df.drop(columns="formula").to_string(index=False))
    print(model_df.drop(columns="thop_breakdown").to_string(index=False))
    for _, r in model_df.iterrows():
        print(f"  {r.model} thop breakdown: {r.thop_breakdown}")

    # Latency is measured on the untrained models; weight values don't change cost.
    lat_rows = []
    for name, model in models.items():
        for row in measure_latency(model, test_loader):
            lat_rows.append({"model": name, **row})
    lat_df = pd.DataFrame(lat_rows)
    lat_df.to_csv(RESULTS / "latency.csv", index=False)
    print(lat_df.to_string(index=False))

    logs, results = [], []
    for name, model in models.items():
        print(f"Training {name}")
        log = train(name, model, train_loader, val_loader)
        logs += log
        test_acc = evaluate(model, test_loader)
        print(f"  {name} test accuracy {test_acc:.4f}")
        lat = lat_df[lat_df.model == name].set_index("batch").median_ms
        m = model_df.set_index("model").loc[name]
        results.append({
            "model": name, "params": int(m.params_torchinfo),
            "hand_flops": int(m.hand_flops), "thop_flops": int(m.thop_flops),
            "latency_b1_ms": lat[1], "latency_b128_ms": lat[128],
            "val_acc": log[-1]["val_acc"], "test_acc": test_acc,
            "train_s_per_epoch": statistics.mean(r["epoch_s"] for r in log),
        })
    pd.DataFrame(logs).to_csv(RESULTS / "training_log.csv", index=False)
    res = pd.DataFrame(results)
    res.to_csv(RESULTS / "results.csv", index=False)
    print(res.to_string(index=False))
    plot_trade_offs(res)


if __name__ == "__main__":
    main()
