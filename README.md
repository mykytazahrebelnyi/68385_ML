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

Datasets are downloaded into `data/`, which is git-ignored.
