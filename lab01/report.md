# Lab 01 — Environment and first system measurements

## 1. Goal

Set up a reproducible Python environment, train two baseline classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset, measure what they cost to run (training time, single-sample latency, model size, peak memory), and decide which deployment tiers (Cloud, Edge, Mobile, TinyML) each model fits.

## 2. Method

**Environment.** Python 3.11.8 in an isolated `.venv`, with dependencies pinned in `requirements.txt`. `src/print_versions.py` writes the interpreter and package versions to `results/versions.txt`. Hardware details are in `results/system_info.txt`: Intel Core (Family 6 Model 165), 6 physical / 12 logical cores, 15.8 GB RAM, Windows 10.

*Deviation from the manual:* the manual pins `mlflow==2.14.1` and `pyarrow==16.1.0`, but these two can't be installed together because mlflow 2.14.1–2.15.1 require `pyarrow<16`. I pinned `mlflow==2.16.0`, the oldest release that works with `pyarrow==16.1.0`. All other packages use exactly the versions the manual lists.

**Data and models.** `load_breast_cancer(return_X_y=True)` gives 569 samples with 30 features. I split it 70/30 (398 train / 171 test), stratified, with `random_state=42`. Seeds for `random`, `numpy` and `torch` are set to 42. The two models use the exact configurations from the manual:

- `LogisticRegression(max_iter=1000, random_state=42)`
- `RandomForestClassifier(n_estimators=100, random_state=42)`

The features are not scaled, so lbfgs reaches the 1000-iteration limit without fully converging. I kept the configuration the manual specifies.

**Measurements** (`src/run_lab01.py`):

- *Training time:* 1 warm-up fit, then 5 timed fits of a fresh clone using `time.perf_counter`. I report the median.
- *Inference latency:* `predict` on one test sample. 1 warm-up, then 100 timed calls. I report the median (p95 is in the CSV).
- *Model size:* `joblib.dump` to `results/model_<name>.joblib`, then the file size on disk. `results/model.joblib` holds the more accurate model.
- *Peak memory:* measured three ways.
  - Peak process RSS, sampled every 1 ms by `memory_profiler` during training and during 100 inferences.
  - Peak heap allocated by the fit/predict call itself, measured with `tracemalloc`.
  - The RSS of the Python + NumPy + pandas + scikit-learn runtime before torch is imported. Torch is imported only to set its seed, and that adds about 123 MB that a deployed scikit-learn model would not need.

## 3. Results

**Accuracy** (`results/baseline_accuracy.csv`)

| Model | Test accuracy |
|---|---|
| LogisticRegression | **0.9415** |
| RandomForest | 0.9357 |

**System cost** (`results/measurements.csv`)

| Metric | LogisticRegression | RandomForest |
|---|---|---|
| Training time, median of 5 (s) | 0.3819 | **0.1454** |
| Single-sample latency, median of 100 (ms) | **0.069** | 2.694 |
| Single-sample latency, p95 (ms) | **0.092** | 3.640 |
| Model size (bytes) | **1,055** | 290,889 |
| Model size (KB) | **1.03** | 284.07 |
| Peak heap allocated during training (KB) | 67.4 | 225.3 |
| Peak heap allocated during one prediction (KB) | 1.7 | 12.9 |
| Peak process RSS, training / inference (MB) | 265.4 / 265.4 | 266.3 / 266.7 |
| Deployment memory: sklearn runtime + model + scratch (MB) | 141.3 | 141.6 |

**Resource budget fit** (`results/budget_fit.csv`). ✅ = within budget, ❌ = over budget.

| Tier | Budget (mem / latency / size) | LogReg latency | LogReg size | LogReg memory | **LogReg fits** | RF latency | RF size | RF memory | **RF fits** |
|---|---|---|---|---|---|---|---|---|---|
| Cloud | ≥ 1 GB / ≤ 100 ms / ≤ 500 MB | ✅ | ✅ | ✅ | **Yes** | ✅ | ✅ | ✅ | **Yes** |
| Edge | 256–1024 MB / ≤ 50 ms / ≤ 50 MB | ✅ | ✅ | ✅ 141 MB | **Yes** | ✅ | ✅ | ✅ 142 MB | **Yes** |
| Mobile | 64–256 MB / ≤ 20 ms / ≤ 10 MB | ✅ | ✅ | ✅ 141 MB | **Yes** | ✅ | ✅ | ✅ 142 MB | **Yes** |
| TinyML | ≤ 256 KB / ≤ 10 ms / ≤ 100 KB | ✅ | ✅ 1 KB | ❌ in Python; ✅ ~3 KB as a C port | **Only if ported** | ✅ | ❌ 284 KB | ❌ ~297 KB even as a C port | **No** |

**Deployment decision**

- **LogisticRegression** can run on all four tiers. It fits Cloud, Edge and Mobile as it is: a 1 KB model, 0.07 ms latency, and about 141 MB of memory with the Python/scikit-learn runtime, which is inside the Mobile range of 64–256 MB. It cannot run on TinyML as a Python program, because the runtime alone is about 550 times the 256 KB limit. The model itself is just 30 weights and a bias, though. Exported as C code, it needs about 1 KB of weights and under 2 KB of scratch memory, and a dot product takes far less than 10 ms even on a microcontroller.
- **RandomForest** fits Cloud, Edge and Mobile. Its latency (2.7 ms median, 3.6 ms p95) is well under the 20 ms Mobile limit, and its 284 KB file is far below 10 MB. It is **not suitable for TinyML**: the serialized forest alone (284 KB) is above both the 100 KB size limit and the 256 KB memory limit. To fit, it would need far fewer or shallower trees.
- Given these numbers, logistic regression is the better choice for any constrained target. It is more accurate on this split (0.9415 vs 0.9357), about 40 times faster per prediction, and about 275 times smaller. The random forest's only advantage is faster training (0.15 s vs 0.38 s), and that does not matter once the model is deployed.

## 4. Conclusions

1. **The runtime takes far more memory than the model.** Both models need less than 300 KB of memory of their own, yet the process holds about 141 MB just to load Python and scikit-learn. Importing torch, which only happens to set its seed, raised that to about 264 MB. So for small models, the deployment tier depends on the runtime and libraries more than on the model. Getting onto TinyML requires a different runtime, such as exported C code, rather than a smaller model.
2. **A more complex model is not automatically better.** The 100-tree forest is about 275 times larger and about 40 times slower per prediction than logistic regression, yet it is slightly *less* accurate on this test set. On a small, almost linearly separable tabular problem, the extra capacity only adds system cost. Measuring cost next to accuracy makes that trade-off visible.
3. **How you measure changes the result.** Single timings varied a lot: logistic regression training ranged up to 0.63 s in an earlier run, against a median of about 0.38 s. That is why I report medians after a warm-up. Sampling memory with RSS completely missed the models' allocations, which were only kilobytes against a process of hundreds of MB, so I needed `tracemalloc` to see them. Every number in this report depends on the method used to get it, and the report has to state that method.


To reproduce, follow the commands in the repository `README.md`.
