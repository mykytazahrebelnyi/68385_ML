"""Lab 01: baseline models on Breast Cancer Wisconsin and their system cost.

Produces in lab01/results/:
  baseline_accuracy.csv  test accuracy of both models
  measurements.csv       training time, latency, size, peak memory
  budget_fit.csv         pass/fail against Cloud / Edge / Mobile / TinyML budgets
  model.joblib           best model (plus model_<name>.joblib for each)
  system_info.txt        hardware / OS the numbers were measured on
"""
import os
import platform
import random
import statistics
import time
import tracemalloc
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
from memory_profiler import memory_usage
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
N_TRAIN_RUNS = 5
N_INFER_RUNS = 100
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Budgets from the lab manual. Memory in MB, latency in ms, size in MB.
BUDGETS = {
    "Cloud": {"mem_mb": float("inf"), "latency_ms": 100, "size_mb": 500},
    "Edge": {"mem_mb": 1024, "latency_ms": 50, "size_mb": 50},
    "Mobile": {"mem_mb": 256, "latency_ms": 20, "size_mb": 10},
    "TinyML": {"mem_mb": 256 / 1024, "latency_ms": 10, "size_mb": 100 / 1024},
}


def set_seeds():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    except ImportError:
        pass


def rss_mb():
    return psutil.Process().memory_info().rss / 2**20


def median_train_time(model, X, y):
    clone(model).fit(X, y)  # warm-up
    times = []
    for _ in range(N_TRAIN_RUNS):
        m = clone(model)
        t0 = time.perf_counter()
        m.fit(X, y)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def median_latency_ms(model, x):
    model.predict(x)  # warm-up
    times = []
    for _ in range(N_INFER_RUNS):
        t0 = time.perf_counter()
        model.predict(x)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), times


def peak_rss(fn):
    """Peak process RSS (MB) sampled by memory_profiler while fn runs."""
    return memory_usage((fn, (), {}), interval=0.001, max_usage=True, include_children=False)


def peak_alloc_kb(fn):
    """Peak Python/NumPy heap allocated by fn itself (tracemalloc), in KB."""
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024


def system_info():
    return "\n".join(
        [
            f"os={platform.platform()}",
            f"python={platform.python_version()}",
            f"cpu={platform.processor()}",
            f"physical_cores={psutil.cpu_count(logical=False)}",
            f"logical_cores={psutil.cpu_count(logical=True)}",
            f"ram_gb={psutil.virtual_memory().total / 2**30:.1f}",
        ]
    )


def main():
    # lbfgs hits max_iter=1000 on unscaled features; the config is fixed by the manual.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    rss_runtime = rss_mb()  # Python + numpy + pandas + sklearn loaded
    set_seeds()
    rss_with_torch = rss_mb()  # torch imported only to seed it
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "system_info.txt").write_text(
        system_info()
        + f"\nrss_sklearn_runtime_mb={rss_runtime:.1f}"
        + f"\nrss_after_torch_import_mb={rss_with_torch:.1f}\n"
    )

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    sample = X_te[:1]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }

    acc_rows, meas_rows, fitted = [], [], {}
    for name, proto in models.items():
        train_s, train_runs = median_train_time(proto, X_tr, y_tr)

        model = clone(proto)
        train_peak = peak_rss(lambda: model.fit(X_tr, y_tr))
        train_alloc = peak_alloc_kb(lambda: clone(proto).fit(X_tr, y_tr))
        fitted[name] = model

        acc = accuracy_score(y_te, model.predict(X_te))
        acc_rows.append({"model": name, "test_accuracy": round(acc, 4)})

        lat_ms, lat_runs = median_latency_ms(model, sample)
        infer_peak = peak_rss(lambda: [model.predict(sample) for _ in range(N_INFER_RUNS)])
        infer_alloc = peak_alloc_kb(lambda: model.predict(sample))

        path = RESULTS / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        meas_rows.append(
            {
                "model": name,
                "train_time_median_s": round(train_s, 4),
                "train_time_min_s": round(min(train_runs), 4),
                "train_time_max_s": round(max(train_runs), 4),
                "latency_median_ms": round(lat_ms, 4),
                "latency_p95_ms": round(float(np.percentile(lat_runs, 95)), 4),
                "model_size_bytes": size_b,
                "model_size_kb": round(size_b / 1024, 2),
                "train_peak_rss_mb": round(train_peak, 1),
                "train_peak_alloc_kb": round(train_alloc, 1),
                "infer_peak_rss_mb": round(infer_peak, 1),
                "infer_peak_alloc_kb": round(infer_alloc, 1),
                # what a deployment needs: sklearn runtime + model in RAM + inference scratch
                "infer_deploy_mem_mb": round(
                    rss_runtime + (size_b / 1024 + infer_alloc) / 1024, 1
                ),
            }
        )

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv(RESULTS / "baseline_accuracy.csv", index=False)
    meas_df = pd.DataFrame(meas_rows)
    meas_df.to_csv(RESULTS / "measurements.csv", index=False)

    best = acc_df.sort_values("test_accuracy", ascending=False).iloc[0]["model"]
    joblib.dump(fitted[best], RESULTS / "model.joblib")

    # Deployment fit. "runtime" = Python+sklearn process without torch plus the model;
    # "model_only" = serialized model + inference scratch, i.e. a port to C/microcontroller.
    fit_rows = []
    for _, r in meas_df.iterrows():
        size_mb = r["model_size_bytes"] / 2**20
        for tier, b in BUDGETS.items():
            lat_ok = r["latency_median_ms"] <= b["latency_ms"]
            size_ok = size_mb <= b["size_mb"]
            mem_proc_ok = r["infer_deploy_mem_mb"] <= b["mem_mb"]
            mem_model_ok = (r["model_size_kb"] + r["infer_peak_alloc_kb"]) / 1024 <= b["mem_mb"]
            fit_rows.append(
                {
                    "model": r["model"],
                    "tier": tier,
                    "latency_ok": lat_ok,
                    "size_ok": size_ok,
                    "memory_ok_python_runtime": mem_proc_ok,
                    "memory_ok_model_only": mem_model_ok,
                    "fits_python_runtime": lat_ok and size_ok and mem_proc_ok,
                    "fits_model_only": lat_ok and size_ok and mem_model_ok,
                }
            )
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(RESULTS / "budget_fit.csv", index=False)

    pd.set_option("display.width", 200)
    print(acc_df.to_string(index=False), end="\n\n")
    print(meas_df.T.to_string(header=False), end="\n\n")
    print(fit_df.to_string(index=False))
    print(f"\nmodel.joblib = {best}")


if __name__ == "__main__":
    main()
