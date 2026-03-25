PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

.PHONY: venv install run clean

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run:
	$(PY) neuralNetwork.py

clean:
	rm -rf $(VENV)
