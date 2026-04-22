"""Quick inspector for o25 npz files - run locally to see available keys."""
import numpy as np, sys, glob

pattern = sys.argv[1] if len(sys.argv) > 1 else '*.npz'
files = sorted(glob.glob(pattern))
if not files:
    print(f"No files matching {pattern}")
    sys.exit(1)

for path in files[:2]:  # inspect first two
    print(f"\n=== {path} ===")
    d = np.load(path, allow_pickle=True)
    for k in d.keys():
        v = d[k]
        arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
        print(f"  {k:40s} shape={str(arr.shape):20s} dtype={arr.dtype}"
              f"  min={arr.flat[0]:.4g}  max={arr.flat[-1]:.4g}")