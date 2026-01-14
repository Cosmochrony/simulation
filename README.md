# Cosmochrony – Simulation

Simulation algorithms and numerical experiments supporting the *Cosmochrony* framework, with a focus on **χ-field relaxation dynamics** and the technical material referenced in the paper appendix (Numerical Methods / Simulation Algorithms).

> Status: research code / evolving. The goal is clarity + reproducibility rather than production-grade APIs.

## What this repository contains

This repository hosts:
- **Numerical implementations** of finite-dimensional approximations of χ-field relaxation dynamics.
- **Exploratory scripts** (parameter sweeps, diagnostics, stability checks, spectral probes).
- **Outputs** used to generate or validate plots/tables referenced in the paper (when applicable).

Conceptually, the simulations are meant as *computational probes* of the Cosmochrony dynamics:
they do **not** introduce additional physical postulates (no fundamental lattice/graph assumption),
but use auxiliary discretizations/bases for numerical stability and diagnostics.

## Relation to the Cosmochrony paper

These simulations support the numerical/technical appendix of the Cosmochrony manuscript,
especially the material around:
- discretized/coarse-grained relaxation flows,
- soliton-like localized configurations and stability regimes,
- spectral / Laplacian-style diagnostics (when used),
- convergence and bounded-variation behavior.

If you are reading the paper: see the appendix section on *Simulation Algorithms for χ-Field Dynamics* (and related technical supplements).

- Paper / main project: https://github.com/Cosmochrony  
- Website (if applicable): https://cosmochrony.org

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
python -m pip install -U pip
```
### 2) Install dependencies
If you have a requirements.txt:
```bash
pip install -r requirements.txt
```
If you don’t yet, the typical minimal stack is:
```bash
pip install numpy scipy matplotlib
```
### 3) Run a simulation script
This repo is script-driven. Typical usage:
```bash
python path/to/script.py --help
python path/to/script.py
```
If a main entrypoint exists (example):
```bash
python main.py
```
If scripts write results, they should do so under output/ (recommended) and/or data/.

Recommended repository layout (target)
If the repo is still organic, this structure keeps it readable:

```python
simulation/
  README.md
  LICENSE
  CITATION.cff
  requirements.txt (or pyproject.toml)
  src/
    cosmochrony_sim/
      __init__.py
      ...
  scripts/
    run_relaxation.py
    sweep_params.py
    diagnostics_spectrum.py
  data/          # optional: small input data, configs
  output/        # generated results (gitignored or partially tracked)
  notebooks/     # optional: exploratory notebooks
  docs/          # optional: extra notes, figures
```
## Outputs and reproducibility
Outputs (plots, CSV, logs) should be generated deterministically when possible.

If randomness is used, scripts should expose a --seed option and log it.

To keep the repo light, large outputs should not be committed (use release assets / Zenodo instead).

## How to cite
If you use this code in academic work, please cite the Cosmochrony paper and (optionally) this repository.
A [CITATION.cff](CITATION.cff) is available to standardize citations.

## Contributing / contact
Issues and PRs are welcome, especially for:

improving reproducibility (CLI options, config files),

documentation and “how to reproduce figure X from the paper” recipes,

performance improvements that preserve clarity.

## License
This repository is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).