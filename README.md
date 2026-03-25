# Neural network (NumPy)

A minimal feedforward neural network with one hidden layer, implemented in NumPy with sigmoid activations. The included demo trains on the XOR problem.

## Setup

Create a virtual environment and install dependencies:

```bash
make install
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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
