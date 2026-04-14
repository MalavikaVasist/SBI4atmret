# SBI4exoplanets

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Checked with MyPy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)

Python package for simulation-based inference (SBI) applied to exoplanet atmospheric retrieval.

## 🚀 Quickstart

Installation:

```bash
git clone <repository-url>
cd SBI4exoplanetsPython
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

## 🏕 Setting up the environment

Set environment variables for data and experiments:

```bash
export SBI4EXOPLANETS_DATA_DIR=/path/to/data
export SBI4EXOPLANETS_EXPERIMENTS_DIR=/path/to/experiments
```

## 🐭 Tests

Run tests with:

```bash
pytest
```

## 📜 Citation

If you use this code, please cite:

[Add citation here]

## ⚖️ License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## 📁 Project Structure

```
SBI4exoplanetsPython/
├── sbi4exoplanets/          # Main package
│   ├── __init__.py
│   ├── train_general.py     # Training script
│   ├── script_general.json  # Configuration
│   ├── utils/               # Utility modules
│   ├── observations/        # Observation data
│   └── simulations/         # Simulation scripts
├── experiments/             # Experiment outputs
├── scripts/                 # Utility scripts
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration
├── README.md
├── LICENSE
└── .gitignore
```
