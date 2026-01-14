# Cosmochrony – Simulation

Simulation algorithms and numerical experiments supporting the *Cosmochrony* framework, with a focus on **χ-field relaxation dynamics** and the technical material referenced in the paper appendix (Numerical Methods / Simulation Algorithms).

> Status: research code / evolving. The goal is clarity and reproducibility rather than production-grade APIs.

## What this repository contains

This repository hosts:
- **Numerical implementations** of finite-dimensional approximations of χ-field relaxation dynamics.
- **Thematic numerical experiments**, organized by scientific axis (spectral analysis, geometry, appendix sweeps).
- **Executable scripts** used to generate figures, tables, and validation results referenced in the manuscript.

Conceptually, the simulations are meant as *computational probes* of the Cosmochrony dynamics:
they do **not** introduce additional physical postulates (no fundamental lattice or graph assumption),
but rely on auxiliary discretizations or bases for numerical stability and diagnostics.

## Repository structure

The repository is organized by **scientific role**, not as a Python package:

```text
simulation/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── scripts/        # Explicit entry points (what to run)
│   ├── main.py
│   ├── toy_cosmochrony_1d.py
│   ├── toy_cosmochrony_1d_a.py
│   ├── chi_relaxation_validation.py
│   ├── critical_tests_cosmochrony.py
│   ├── collect_D4_csv.py
│   ├── galaxy_rotcurve.py
│   ├── galaxy_rotcurves_3panel.py
│   ├── plot_cmb_lowell_planck_vs_cosmochrony.py
│   └── cmb_lowell_tests_corr_rescale.py
│
├── spectral/       # Spectral / Laplacian diagnostics (eigenmodes, ratios, convergence)
│   ├── lap_ratio.py
│   ├── spectral_ratio.py
│   ├── convergence_8_3.py
│   ├── eigenmodes.py
│   ├── spectral_test.py
│   └── make_spectral_fig.py
│
├── geometry/       # S³, Hopf fiberbase, weighted Laplacians, stiffness / curvature
│   ├── compare_mc_vs_weighted_laplacian_hopf_*.py
│   ├── compare_mc_vs_weighted_laplacian_s3.py
│   ├── weighted_laplacian_s3_bias.py
│   ├── weighted_measure_laplacian_s3_fiberbase.py
│   ├── weighted_laplacian_s3_fiberbase_db.py
│   ├── s3_relaxation_bias_ratio.py
│   ├── curvature_derivation.py
│   ├── stiffness_derivation.py
│   ├── stiffness_integration.py
│   └── stiffness_ratio.py
│
├── appendix_D4/    # Appendix D4 numerical sweeps and aggregated results
│   ├── summary_D4_all.csv
│   └── sweeps/
│
├── data/           # Input data (Planck CMB, galaxy rotation curves)
│   ├── Planck/
│   └── Rotmod_LTG/
│
├── figures/        # Final figures referenced in the manuscript
└── output/         # Optional: transient/generated outputs (typically gitignored)
```
## Relation to the Cosmochrony paper

These simulations support the numerical and technical appendices of the Cosmochrony manuscript,
in particular:
- discretized and coarse-grained relaxation flows,
- stability of localized (soliton-like) configurations,
- spectral / Laplacian diagnostics (e.g. convergence and ratio tests),
- Appendix D4 sweeps and saturation analyses,
- comparative studies (CMB low-ℓ, galaxy rotation curves).

If you are reading the paper, see the appendix section Simulation Algorithms for χ-Field Dynamics and related technical supplements.
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
```bash
pip install -r requirements.txt
```
### 3) Run a simulation script
This repository is script-driven. Typical usage:
```bash
python scripts/<script>.py
python spectral/<script>.py
python geometry/<script>.py
```
Examples:
```bash
python scripts/main.py
python spectral/convergence_8_3.py
python scripts/galaxy_rotcurves_3panel.py
python scripts/plot_cmb_lowell_planck_vs_cosmochrony.py
```

## Outputs and reproducibility
- Many scripts generate figures (PDF/PNG) directly in figures/, appendix_D4/, or the repository root.
- When randomness is involved, scripts are expected to expose a --seed parameter or log the seed explicitly.
- Large sweep outputs are grouped under appendix_D4/ for traceability with the manuscript.

## How to cite
If you use this code in academic work, please cite the Cosmochrony paper.
A [CITATION.cff](CITATION.cff) is provided to standardize citations.
(preferred citation points to the Zenodo DOI of the manuscript).

## Contributing / contact
Issues and pull requests are welcome, especially for:
- improving reproducibility (CLI options, configuration files),
- documentation and “how to reproduce figure X from the paper” recipes,
- numerical robustness and clarity of the experiments.

## License
This repository is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).