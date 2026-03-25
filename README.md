# Neural network (NumPy)

A minimal feedforward neural network with one hidden layer, implemented in NumPy with sigmoid activations. The included demo trains on the XOR problem.

## Setup

Pick **one** of the following (do **not** paste the word `or` into the terminal — that is only English “choose A or B”).

**Option A — Makefile** (creates `.venv` if missing, then installs packages):

```bash
make install
```

**Option B — manual** (same result, no `make`):

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\pip` instead of `.venv/bin/pip`, or activate first: `.venv\Scripts\activate`.

`requirements.txt` includes TensorFlow and Keras; the install is large and can take several minutes.

## Run

```bash
make run
```

Or:

```bash
python neuralNetwork.py
```

After training, the script prints rounded predictions for the four XOR inputs; you should see values close to `0` or `1`.

## Project layout

| File | Purpose |
|------|---------|
| `neuralNetwork.py` | Network class and XOR demo |
| `requirements.txt` | Python dependencies |
| `Makefile` | `install`, `run`, `clean` |

## Requirements

- Python 3.10+ recommended
- NumPy (see `requirements.txt`)
