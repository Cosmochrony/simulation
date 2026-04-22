"""
Diagnostic: inspect the stored vecs_c/vecs_qmc in an O25 npz file
produced with --store-vectors, and run Test 4 with varying thresholds.

Usage: python inspect_vecs.py ./o25_outputs/q29_o25.npz
"""
import sys, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else './o25_outputs/q29_o25.npz'
d = np.load(path, allow_pickle=True)

if 'vecs_c' not in d:
    print("No vecs_c -- re-run with --store-vectors --force")
    sys.exit(0)

vc   = d['vecs_c']
vqmc = d['vecs_qmc']
n0, n1 = int(d['n0']), int(d['n1'])
q = int(d['q'])
P = len(d['pairs'])
sigma_c = d['sigma_c_mean']      # (P, N_shells)
ns = d['ns']

print(f"q={q}, n0={n0}, n1={n1}, P={P}, vecs shape={vc.shape}")
print()

# Show distribution of residual norms
all_norms = []
for i in range(P):
    for k in range(vc.shape[1]):
        v = vc[i, k]
        if v is None: continue
        v = np.asarray(v, dtype=np.float64)
        all_norms.extend(v[v > 0].tolist())

if all_norms:
    a = np.array(all_norms)
    print(f"Non-zero residual norms (all pairs, all shells):")
    print(f"  count   = {len(a)}")
    print(f"  min     = {a.min():.4e}")
    print(f"  median  = {np.median(a):.4e}")
    print(f"  max     = {a.max():.4e}")
    print(f"  > 1e-6  = {np.sum(a > 1e-6)}")
    print(f"  > 1e-8  = {np.sum(a > 1e-8)}")
    print(f"  > 1e-10 = {np.sum(a > 1e-10)}")
    print()

# Compare sigma_c from stored vecs vs from sigma_c_mean
print("Sanity check: ||v_c^(n)||^2 vs sigma_c(n) * |S_n|")
idx = np.where((ns >= n0) & (ns <= n1))[0]
for i in range(min(3, P)):
    c, qc = d['pairs'][i]
    for k, ni in enumerate(idx[:3]):
        v = vc[i, k]
        if v is None: continue
        v = np.asarray(v, dtype=np.float64)
        norm_sq = float(np.dot(v, v))
        sc_expected = float(sigma_c[i, ni]) * int(d['shell_sizes'][ni])
        print(f"  pair({c},{qc}) shell n={ns[ni]}: "
              f"||v||^2={norm_sq:.4e}  sigma_c*|S_n|={sc_expected:.4e}")
print()

# Test 4B with different thresholds
print("Test 4B r_eff vs threshold:")
for thr in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
    reff_list = []
    for i in range(P):
        valid = []
        for k in range(vc.shape[1]):
            v  = np.asarray(vc[i, k]   if vc[i, k]   is not None else [], dtype=np.float64)
            vq = np.asarray(vqmc[i, k] if vqmc[i, k] is not None else [], dtype=np.float64)
            if len(v) > 0 and len(vq) > 0:
                valid.append((v, vq))
        if not valid:
            reff_list.append(0)
            continue
        d_c   = min(len(v)  for v, _  in valid)
        d_qmc = min(len(vq) for _, vq in valid)
        vecs = [np.outer(v[:d_c], vq[:d_qmc]).ravel() for v, vq in valid]
        V = np.stack(vecs, axis=0)
        G = (V @ V.T) / len(vecs)
        sv = np.linalg.svd(G, compute_uv=False)
        r = int(np.sum(sv > thr * sv[0])) if sv[0] > 0 else 0
        reff_list.append(r)
    from collections import Counter
    c = Counter(reff_list)
    print(f"  threshold={thr:.0e}: r_eff distribution = {dict(sorted(c.items()))}")
