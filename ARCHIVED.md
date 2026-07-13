# simulation — ARCHIVED

This repository has been **archived**. Its Python scripts were relocated into the paper repositories that own the
result each script establishes or audits, each script now living in the `code/` subdirectory of its target repo,
with git history preserved (via `format-patch` / `am`).

See `../SIMULATION-RELOCATION-STATUS.md` for the full mapping, method, and per-repo status.

## What remains here

- No scripts remain: all Python scripts have been relocated (including the last two into `white-paper/code/` and
  `white-paper/appendix-d4/`).
- Datasets, which were **copied** (not moved) to the relevant repos and are kept here as the archived source of record:
  - `data/Planck`, `data/Rotmod_LTG` → also in `cosmology-paper/data/`
  - `appendix_D4/`, `D-appendix-technical/`, `figures/*D4*` → also in `white-paper/appendix-d4/`

## Note

The relocated copies of shared modules (`spectral_O12.py`, `o25_paired_pipeline.py`) intentionally diverge per
experiment; deduplication into a shared package is a separate, future refactoring and was deliberately not done as
part of the relocation.
