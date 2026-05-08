"""Quick inspector for o25 npz files - run locally to see available keys."""
import numpy as np, sys, glob

pattern = sys.argv[1] if len(sys.argv) > 1 else '*.npz'
files = sorted(glob.glob(pattern))
if not files:
    print(f"No files matching {pattern}")
    sys.exit(1)

for path in files[:2]:
    print(f"\n=== {path} ===")
    d = np.load(path, allow_pickle=True)
    for k in d.keys():
        v = d[k]
        arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
        if arr.dtype == object:
            # Object array: show shape and type of first non-None element
            first = next((x for x in arr.flat if x is not None), None)
            inner = f"inner={type(first).__name__}{getattr(first,'shape','')}" if first is not None else "all None"
            print(f"  {k:40s} shape={str(arr.shape):20s} dtype=object  [{inner}]")
        else:
            try:
                mn = float(arr.flat[0])
                mx = float(arr.flat[-1])
                print(f"  {k:40s} shape={str(arr.shape):20s} dtype={arr.dtype}"
                      f"  first={mn:.4g}  last={mx:.4g}")
            except Exception:
                print(f"  {k:40s} shape={str(arr.shape):20s} dtype={arr.dtype}")