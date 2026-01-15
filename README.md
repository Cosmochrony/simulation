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

## Reproducing results

This section provides minimal, concrete recipes to reproduce representative numerical
results used in the Cosmochrony manuscript.  
All commands assume that the virtual environment is activated and dependencies installed.

---

### 1) Appendix D4 — χ-field relaxation sweeps and saturation diagnostics

Appendix D4 relies on parameter sweeps whose aggregated results are stored under
`appendix_D4/`.

To regenerate or aggregate sweep summaries:

```bash
python scripts/collect_D4_csv.py
```
This script scans the sweep result directories and produces consolidated CSV summaries
(e.g. summary_D4_all.csv), used to generate the figures in Appendix D4.

Precomputed sweep outputs and figures are available under:
```bath
appendix_D4/sweeps/
```
### 2) Spectral diagnostics — convergence and ratio tests

Key spectral diagnostics (including convergence and characteristic ratios) are implemented
in the `spectral/` directory.

Typical runs:
```bash
python spectral/convergence_8_3.py
python spectral/spectral_ratio.py
python spectral/lap_ratio.py
```
To generate the corresponding spectral figures:
```bash
python spectral/make_spectral_fig.py
```
These scripts probe eigenmode structure, convergence behavior, and robustness of the
spectral ratios discussed in the manuscript.

### 3) CMB low-ℓ comparison (Planck vs Cosmochrony)
Low-ℓ CMB comparisons use publicly available Planck Release 3 data included under `data/Planck/`.

To generate the comparison plots:
```bash
python scripts/plot_cmb_lowell_planck_vs_cosmochrony.py
```
Additional rescaling / correction tests:
```bash
python scripts/cmb_lowell_tests_corr_rescale.py
```
Generated figures include:
- `cmb_lowell_planck_lcdm_cosmochrony.pdf`
- related diagnostic PDFs in the repository root or `figures/`.

### 4) Galaxy rotation curves

Galaxy rotation curve data (LTG sample) are provided under `data/Rotmod_LTG/`.

To reproduce the multi-panel rotation curve comparison:
```bash
python scripts/galaxy_rotcurves_3panel.py
```
This generates:
- `galaxy_rotcurves_3panel.pdf`

### Notes on reproducibility
- Scripts are intended to be **self-contained and explicit**, rather than hidden behind a unified pipeline.
- Many scripts generate figures directly as PDF/PNG for traceability with the manuscript.
- When randomness is involved, scripts either fix or report the random seed.
- Large parameter sweeps are precomputed and versioned for transparency.

### Scope and intent
These simulations are numerical probes, not a standalone simulation framework.
Their purpose is to:
- validate internal consistency,
- explore stability and convergence regimes,
- support figures and tables in the Cosmochrony manuscript.

They should be read in conjunction with the corresponding theoretical sections of the paper.

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