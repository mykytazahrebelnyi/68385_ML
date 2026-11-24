# 68385_ML

Machine Learning Systems labs. Author: Mykyta Zahrebelnyi (68385).

## Setup

Python 3.11.8:

```
python3.11 -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`mlflow` is pinned to 2.16.0 instead of the 2.14.1 the manual lists, because mlflow 2.14.1 requires `pyarrow<16`, which conflicts with the required `pyarrow==16.1.0`.

## Labs

| Lab | Topic | Run |
|---|---|---|
| [lab01](lab01/report.md) | Environment and first system measurements | `python lab01/src/print_versions.py` then `python lab01/src/run_lab01.py` |
| [lab02](lab02/report.md) | A neural network from scratch (NumPy MLP, backprop, gradient check) | `python lab02/src/run_lab02.py` |
| [lab03](lab03/report.md) | Architectures and their costs (MLP vs CNN vs attention, FLOPs, latency) | `python lab03/src/run_lab03.py` |

Datasets are downloaded into `data/`, which is git-ignored.
