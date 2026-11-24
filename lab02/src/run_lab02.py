"""Lab 02: a two-hidden-layer neural network in NumPy, from scratch.

64 -> W -> ReLU -> W -> ReLU -> 10 -> softmax, trained on load_digits.

Produces in lab02/results/:
  sanity_checks.txt      forward-pass shape / row-sum checks and initial loss
  gradient_check.csv     analytic vs numerical slope for 10 random weights
  training_log.csv       per-epoch train/test loss and accuracy, every run
  final_accuracy.csv     final test accuracy per (width, learning rate)
  loss_curves.png        loss curves for all 6 runs
  timing.csv             forward vs backward time, median of 100 runs
"""
import os
import platform
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
WIDTHS = [16, 64]
LEARNING_RATES = [0.001, 0.01, 0.1]
BATCH_SIZE = 32
EPOCHS = 50
INIT_STD = 0.01  # N(0, 0.01), read as standard deviation 0.01
GRAD_EPS = 1e-5
GRAD_N_WEIGHTS = 10
GRAD_TOL = 1e-6
N_TIMING_RUNS = 100
TIMING_WIDTHS = WIDTHS + [256, 1024]  # wider nets only for timing, to show the trend
RESULTS = Path(__file__).resolve().parents[1] / "results"
WEIGHT_NAMES = ["W1", "W2", "W3"]


# ---------------------------------------------------------------- data (task 1)
def load_data():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float64) / 16.0
    Y = np.eye(10)[y]
    return train_test_split(X, Y, test_size=0.3, stratify=y, random_state=SEED)


# ------------------------------------------------------------- model (task 2)
def init_params(width, rng):
    shapes = {
        "W1": (64, width), "b1": (width,),
        "W2": (width, width), "b2": (width,),
        "W3": (width, 10), "b3": (10,),
    }
    return {
        name: rng.normal(0.0, INIT_STD, shape) if name.startswith("W") else np.zeros(shape)
        for name, shape in shapes.items()
    }


def n_params(params):
    return sum(p.size for p in params.values())


# ------------------------------------------------------- forward (tasks 3, 4)
def relu(z):
    return np.maximum(z, 0.0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, X):
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)
    z3 = a2 @ params["W3"] + params["b3"]
    p = softmax(z3)
    cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "p": p}
    return p, cache


def cross_entropy(p, Y):
    # Mean over the batch of -sum_k y_k log p_k. The clip only guards log(0).
    return float(-np.mean(np.sum(Y * np.log(np.clip(p, 1e-12, None)), axis=1)))


# ----------------------------------------------------------- backward (task 5)
def backward(params, cache, Y):
    """Gradients of the mean cross-entropy w.r.t. every parameter.

    Softmax + cross-entropy together give dL/dz3 = (p - y) / N.
    Linear layer z = a W + b:  dW = a^T dz,  db = sum_rows(dz),  da = dz W^T.
    ReLU:  dz = da * 1[z > 0].
    """
    n = Y.shape[0]
    dz3 = (cache["p"] - Y) / n
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)
    da2 = dz3 @ params["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ params["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = cache["X"].T @ dz1
    db1 = dz1.sum(axis=0)
    # No gradient w.r.t. the input X: nothing upstream needs it.
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def loss_at(params, X, Y):
    p, _ = forward(params, X)
    return cross_entropy(p, Y)


def gradient_check(params, X, Y, rng, label):
    _, cache = forward(params, X)
    grads = backward(params, cache, Y)
    # Draw 10 weights uniformly from the weight-matrix entries whose analytic
    # gradient is non-zero. Zero-gradient entries (W1 rows for pixels that are
    # always blank, columns of dead ReLUs) give 0 == 0 and test nothing.
    candidates = [(name, idx) for name in WEIGHT_NAMES
                  for idx in zip(*np.nonzero(grads[name]))]
    picks = rng.choice(len(candidates), size=GRAD_N_WEIGHTS, replace=False)
    rows = []
    for pick in picks:
        name, idx = candidates[pick]
        original = params[name][idx]
        params[name][idx] = original + GRAD_EPS
        loss_plus = loss_at(params, X, Y)
        params[name][idx] = original - GRAD_EPS
        loss_minus = loss_at(params, X, Y)
        params[name][idx] = original
        numerical = (loss_plus - loss_minus) / (2 * GRAD_EPS)
        analytic = grads[name][idx]
        abs_err = abs(analytic - numerical)
        rel_err = abs_err / max(abs(analytic), abs(numerical))
        rows.append({
            "state": label, "weight": f"{name}{list(map(int, idx))}",
            "analytic": analytic, "numerical": numerical,
            "abs_error": abs_err, "rel_error": rel_err, "pass": rel_err < GRAD_TOL,
        })
    return rows


# ------------------------------------------------------------ training (task 6)
def accuracy(params, X, Y):
    p, _ = forward(params, X)
    return float(np.mean(p.argmax(axis=1) == Y.argmax(axis=1)))


def train(width, lr, X_tr, Y_tr, X_te, Y_te):
    rng = np.random.default_rng(SEED)
    params = init_params(width, rng)  # same init for every lr at a given width
    log = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_tr))
        batch_losses, batch_sizes = [], []
        for start in range(0, len(X_tr), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            p, cache = forward(params, X_tr[idx])
            batch_losses.append(cross_entropy(p, Y_tr[idx]))
            batch_sizes.append(len(idx))
            grads = backward(params, cache, Y_tr[idx])
            for k in params:
                params[k] -= lr * grads[k]
        row = {
            "width": width, "lr": lr, "epoch": epoch,
            "train_loss": float(np.average(batch_losses, weights=batch_sizes)),
            "test_loss": loss_at(params, X_te, Y_te),
            "train_acc": accuracy(params, X_tr, Y_tr),
            "test_acc": accuracy(params, X_te, Y_te),
        }
        log.append(row)
        print(f"  W={width:<3} lr={lr:<6} epoch {epoch:>2}  "
              f"train_loss={row['train_loss']:.4f}  test_loss={row['test_loss']:.4f}  "
              f"test_acc={row['test_acc']:.4f}")
    return params, log


def plot_loss_curves(log_df):
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(12, 4.5), sharey=True)
    colors = dict(zip(LEARNING_RATES, ["tab:blue", "tab:orange", "tab:green"]))
    for ax, width in zip(axes, WIDTHS):
        for lr in LEARNING_RATES:
            run = log_df[(log_df.width == width) & (log_df.lr == lr)]
            ax.plot(run.epoch, run.train_loss, color=colors[lr], label=f"lr={lr} train")
            ax.plot(run.epoch, run.test_loss, color=colors[lr], ls="--", label=f"lr={lr} test")
        ax.axhline(np.log(10), color="grey", lw=0.8, ls=":", label="ln(10)")
        ax.set_title(f"Hidden width W = {width}")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Loss curves: 64-W-W-10 MLP on load_digits, batch 32, SGD")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curves.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- timing (task 7)
def matmul_flops(width, batch):
    # 2*m*k*n per (m,k)@(k,n) matmul.
    fwd = 2 * batch * (64 * width + width * width + width * 10)
    # Backward: dW for all three layers + da for layers 3 and 2 (none for X).
    bwd = 2 * batch * (64 * width + width * width + width * 10) \
        + 2 * batch * (10 * width + width * width)
    return fwd, bwd


def time_passes(params, X, Y, width, batch):
    for _ in range(10):  # warm-up
        _, cache = forward(params, X)
        backward(params, cache, Y)
    fwd_t, bwd_t = [], []
    for _ in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        _, cache = forward(params, X)
        t1 = time.perf_counter()
        backward(params, cache, Y)
        t2 = time.perf_counter()
        fwd_t.append(t1 - t0)
        bwd_t.append(t2 - t1)
    fwd_ms = statistics.median(fwd_t) * 1e3
    bwd_ms = statistics.median(bwd_t) * 1e3
    fwd_flops, bwd_flops = matmul_flops(width, batch)
    return {
        "width": width, "batch": batch,
        "forward_ms": fwd_ms, "backward_ms": bwd_ms,
        "backward_over_forward": bwd_ms / fwd_ms,
        "forward_matmul_flops": fwd_flops, "backward_matmul_flops": bwd_flops,
        "flop_ratio": bwd_flops / fwd_flops,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    X_tr, X_te, Y_tr, Y_te = load_data()
    print(f"train {X_tr.shape}  test {X_te.shape}  x range [{X_tr.min()}, {X_tr.max()}]")

    # Tasks 3 + 4: forward-pass and initial-loss sanity checks.
    lines = [f"train X {X_tr.shape}, Y {Y_tr.shape}; test X {X_te.shape}, Y {Y_te.shape}"]
    for width in WIDTHS:
        params = init_params(width, np.random.default_rng(SEED))
        p, _ = forward(params, X_tr)
        max_dev = float(np.abs(p.sum(axis=1) - 1).max())
        loss0 = cross_entropy(p, Y_tr)
        lines.append(
            f"W={width}: params={n_params(params)}  output shape={p.shape}  "
            f"max |row sum - 1| = {max_dev:.2e} (<1e-6: {max_dev < 1e-6})  "
            f"initial loss = {loss0:.4f} (ln10 = {np.log(10):.4f})"
        )
        assert p.shape == (len(X_tr), 10) and max_dev < 1e-6
    (RESULTS / "sanity_checks.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Task 5: gradient check at init and after a short training run (W=16).
    gc_rng = np.random.default_rng(SEED)
    X_gc, Y_gc = X_tr[:64], Y_tr[:64]
    params = init_params(16, np.random.default_rng(SEED))
    gc_rows = gradient_check(params, X_gc, Y_gc, gc_rng, "init")

    # Task 6: training sweep.
    logs, finals, trained_params = [], [], {}
    for width in WIDTHS:
        for lr in LEARNING_RATES:
            print(f"Training W={width}, lr={lr}")
            params, log = train(width, lr, X_tr, Y_tr, X_te, Y_te)
            logs.extend(log)
            trained_params[(width, lr)] = params
            finals.append({
                "width": width, "lr": lr, "n_params": n_params(params),
                "final_train_loss": log[-1]["train_loss"],
                "final_test_loss": log[-1]["test_loss"],
                "final_train_acc": log[-1]["train_acc"],
                "final_test_acc": log[-1]["test_acc"],
            })
    log_df = pd.DataFrame(logs)
    log_df.to_csv(RESULTS / "training_log.csv", index=False)
    pd.DataFrame(finals).to_csv(RESULTS / "final_accuracy.csv", index=False)
    plot_loss_curves(log_df)

    # Gradient check again on a trained network, where weights and gradients are larger.
    for width in WIDTHS:
        gc_rows += gradient_check(trained_params[(width, 0.1)], X_gc, Y_gc, gc_rng,
                                  f"trained W={width} lr=0.1")
    gc_df = pd.DataFrame(gc_rows)
    gc_df.to_csv(RESULTS / "gradient_check.csv", index=False)
    print(gc_df.to_string(index=False))
    print(gc_df.groupby("state")["pass"].sum())

    # Task 7: forward vs backward timing.
    timing = []
    for width in TIMING_WIDTHS:
        # Values don't affect cost, so a fresh init is fine for every width.
        params = init_params(width, np.random.default_rng(SEED))
        for batch in [BATCH_SIZE, len(X_te)]:
            timing.append(time_passes(params, X_te[:batch], Y_te[:batch], width, batch))
    timing_df = pd.DataFrame(timing)
    timing_df.to_csv(RESULTS / "timing.csv", index=False)
    print(timing_df.to_string(index=False))
    print(f"{platform.processor()} | {os.cpu_count()} logical cores | numpy {np.__version__}")


if __name__ == "__main__":
    main()
