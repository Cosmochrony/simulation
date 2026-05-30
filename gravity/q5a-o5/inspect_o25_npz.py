"""
inspect_o25_npz.py

Inspect the contents of q<q>_o25.npz files to determine what per-block
data is available for the [O-2] numerical test.

For each file, reports:
  - All array keys with shapes and dtypes
  - Whether per-block sigma trajectories are present
  - Whether delta_pair_samples (M exponent samples per pair) are present
  - Whether vecs_c (O26 residual-norm vectors) are present
  - Memory footprint estimates

Usage:
    python inspect_o25_npz.py --dir o25_outputs
    python inspect_o25_npz.py --files q61_o25.npz q151_o25.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


_BYTES_PER_UNIT = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}


def _fmt_size(n_bytes: int) -> str:
    for unit in ('GB', 'MB', 'KB', 'B'):
        if n_bytes >= _BYTES_PER_UNIT[unit]:
            return f'{n_bytes / _BYTES_PER_UNIT[unit]:.2f} {unit}'
    return f'{n_bytes} B'


def inspect_file(path: pathlib.Path) -> None:
    print(f'\n{"=" * 68}')
    print(f'  {path.name}  ({_fmt_size(path.stat().st_size)})')
    print(f'{"=" * 68}')

    z = np.load(path, allow_pickle=True)
    keys = sorted(z.files)

    q = int(z['q']) if 'q' in keys else '?'
    M = int(z['M_per_pair']) if 'M_per_pair' in keys else '?'
    n0 = int(z['n0']) if 'n0' in keys else '?'
    n1 = int(z['n1']) if 'n1' in keys else '?'
    print(f'  q={q}  M_per_pair={M}  window=[{n0},{n1}]')
    print()

    # Classify keys
    per_block = []       # M trajectories per pair (what we want for O-2)
    mean_profiles = []   # M-averaged profiles
    exponent_data = []   # delta_pair exponents
    vectors_o26 = []     # O26 residual-norm vectors (--store-vectors)
    other = []

    for k in keys:
        arr = z[k]
        shape = arr.shape if hasattr(arr, 'shape') else '(scalar)'
        dtype = arr.dtype if hasattr(arr, 'dtype') else type(arr).__name__
        # Estimate memory
        if hasattr(arr, 'nbytes'):
            mem = arr.nbytes
        else:
            mem = 0
        mem_str = _fmt_size(mem) if mem else ''

        row = f'    {k:<30}  {str(shape):<25}  {str(dtype):<12}  {mem_str}'

        if k in ('sigma_c_mean', 'sigma_qmc_mean', 'sigma_pair_mean'):
            mean_profiles.append(row)
        elif k in ('delta_pair_samples', 'r2_samples'):
            exponent_data.append(row)
        elif k.startswith('vecs_') or k.startswith('basis_') or k.startswith('pi_'):
            vectors_o26.append(row)
        elif k.startswith('adm_w_'):
            other.append(row)
        else:
            other.append(row)

    if mean_profiles:
        print('  [M-averaged profiles — available]')
        for r in mean_profiles:
            print(r)
        print()

    if exponent_data:
        print('  [Per-block exponent samples — available]')
        for r in exponent_data:
            print(r)
        print()

    if vectors_o26:
        print('  [O26 residual-norm vectors (--store-vectors) — first block only]')
        for r in vectors_o26:
            print(r)
        # For object arrays, peek at actual content
        if 'vecs_c' in keys:
            vc = z['vecs_c']
            if vc.dtype == object and vc.size > 0:
                # Peek at first non-empty element
                sample = vc.flat[0]
                print(f'    vecs_c[0,0] sample shape: {np.asarray(sample).shape}  '
                      f'(length = shell size at n0)')
        print()

    if other:
        print('  [Other arrays]')
        for r in other:
            print(r)
        print()

    # Summary for [O-2]
    print('  [O-2] DATA AVAILABILITY SUMMARY')
    has_delta_samples = 'delta_pair_samples' in keys
    has_sigma_c_sq = 'sigma_c_sq_mean' in keys  # would store E[sigma_c^2]
    has_sigma_c_all = any(k.startswith('sigma_c_all') for k in keys)
    has_vecs = 'vecs_c' in keys

    print(f'    sigma_c_mean (M-averaged):         YES — shape {z["sigma_c_mean"].shape}')
    print(f'    delta_pair_samples (M exponents):  {"YES" if has_delta_samples else "NO"}'
          + (f' — shape {z["delta_pair_samples"].shape}' if has_delta_samples else ''))
    print(f'    sigma_c_sq_mean (for Var intra):   {"YES" if has_sigma_c_sq else "NO (not stored)"}')
    print(f'    sigma_c_all (M full trajectories): {"YES" if has_sigma_c_all else "NO (not stored)"}')
    print(f'    vecs_c (O26 vectors, block 0):     {"YES" if has_vecs else "NO"}')
    print()

    if has_delta_samples:
        dps = z['delta_pair_samples']
        n_pairs, M_actual = dps.shape
        std_delta = np.nanstd(dps, axis=1)
        print(f'    delta_pair_samples: {n_pairs} pairs × {M_actual} blocks')
        print(f'      std(delta_c) per pair: mean={std_delta.mean():.4f}  '
              f'median={np.nanmedian(std_delta):.4f}  max={std_delta.max():.4f}')
        print(f'      (captures exponent variability; proxy for H(delta_c|c) only,')
        print(f'       NOT H(sigma_c(ell)|c) at each depth)')

    if has_vecs:
        print()
        print('  NOTE: vecs_c contains residual-norm vectors from block 0 ONLY,')
        print('  not M block trajectories. Cannot reconstruct per-block sigma(ell).')
        print('  To test [O-2], either:')
        print('    (a) add sigma_c_sq_mean accumulation to _compute_one_pair (5 lines),')
        print('        rerun pipeline; OR')
        print('    (b) use delta_pair_samples as a surrogate for within-c entropy')
        print('        (tests H(delta|c) not H(sigma(ell)|c), lower information).')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', type=pathlib.Path, default=None,
                        help='Directory containing q*_o25.npz files.')
    parser.add_argument('--files', type=pathlib.Path, nargs='+', default=None,
                        help='Explicit list of .npz files.')
    parser.add_argument('--q-list', type=int, nargs='+', default=None,
                        help='Restrict to specific primes when using --dir.')
    args = parser.parse_args()

    if args.files:
        paths = args.files
    elif args.dir:
        if args.q_list:
            paths = [args.dir / f'q{q}_o25.npz' for q in args.q_list]
        else:
            paths = sorted(args.dir.glob('q*_o25.npz'))
    else:
        parser.error('Either --dir or --files must be specified.')

    for p in paths:
        if not p.exists():
            print(f'[WARN] {p} not found, skipping')
            continue
        inspect_file(p)
    print()


if __name__ == '__main__':
    main()