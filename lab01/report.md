# Lab 01 — Environment and first system measurements

I set up a Python 3.11 virtual environment with the package versions from the manual.  
On the Breast Cancer dataset (70/30 stratified split, seed 42) I trained a logistic regression and a random forest. For each one I measured accuracy, training time, latency per prediction, model size and memory. Everything is in `src/run_lab01.py`, and the numbers are in `results/`.

| | Accuracy | Train time | Latency (1 sample) | Model size |
|---|---|---|---|---|
| Logistic regression | 0.9415 | 0.38 s | 0.07 ms | 1 KB |
| Random forest | 0.9357 | 0.15 s | 2.7 ms | 284 KB |

Logistic regression came out ahead on almost everything: it was slightly more accurate, about 40× faster per prediction and about 275× smaller. Both models fit the Cloud, Edge and Mobile budgets. Neither fits TinyML as Python code, because the Python + scikit-learn runtime alone takes about 141 MB. Logistic regression could run on TinyML if exported to C, since it's just 30 weights. The random forest couldn't, because its 284 KB file is already over the 100 KB limit.

My main takeaway is that for small models the runtime matters more than the model itself. I also learned that how you measure changes the numbers: single timings were noisy, so I used medians after a warm-up. Process memory (RSS) was too coarse to see the models at all, so I measured their allocations with `tracemalloc`.
