"""
check_npz.py - Diagnostic: what is stored in an O25 npz file?
Usage: python check_npz.py ./o25_outputs/q61_o25.npz
"""
import sys, numpy as np
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./o25_outputs/q61_o25.npz')
d = np.load(path, allow_pickle=True)

print(f"\nFile: {path}")
print(f"Keys: {list(d.keys())}")
print(f"q={d['q']}, n0={d['n0']}, n1={d['n1']}, P={len(d['pairs'])}")

for key in ['vecs_c', 'vecs_qmc', 'basis_c', 'basis_qmc', 'pi_c', 'pi_qmc']:
    if key not in d:
        print(f"\n{key}: NOT PRESENT")
        continue
    v = d[key]
    print(f"\n{key}: shape={v.shape}, dtype={v.dtype}")
    # Check first entry
    first = v.flat[0]
    if first is None:
        print(f"  [0,0] = None")
    else:
        arr = np.asarray(first)
        print(f"  [0,0]: shape={arr.shape}, dtype={arr.dtype}, "
              f"norm={float(np.linalg.norm(arr)):.4e}")
        if len(arr) > 0 and np.any(arr != 0):
            print(f"  [0,0] values: {arr[:5]}")
        else:
            print(f"  [0,0] is ZERO or EMPTY")

# Verdict
has_pi = 'pi_c' in d
if has_pi:
    v = d['pi_c']
    first = np.asarray(v.flat[0])
    rank = len(first) if first is not None else 0
    norm = float(np.linalg.norm(first)) if first is not None and len(first)>0 else 0
    if rank > 0 and norm > 1e-15:
        print(f"\nVERDICT: pi_c present, rank={rank}, norm={norm:.4e}")
        print("  -> Run test_reff_phi_o27.py to get r_eff")
    else:
        print(f"\nVERDICT: pi_c present but EMPTY (rank={rank}, norm={norm:.2e})")
        print("  -> Workers used OLD spectral_O12.py")
        print("  -> Replace spectral_O12.py with the new version and re-run:")
        print("     python o25_paired_pipeline.py --primes 29 61 --M 50 \\")
        print("       --bfs-frac 0.99 --store-vectors --force")
else:
    print("\nVERDICT: pi_c NOT PRESENT")
    print("  -> Re-run with --store-vectors --force")
