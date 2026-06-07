"""Alignment proof on the REAL engine: |{m_j > n}| == final_rank - cumsum(delta_r)."""
import numpy as np
import spectral_O12 as o12
from o25_entanglement import adm_entropy_from_admitted

q = 61
gens = o12.build_generators(q)
shells = o12.bfs_shells(None, None, gens, q, 0.99)
rng = np.random.default_rng(123)
c_block = o12.sample_generic_blocks(q, 1, rng)[0]   # one generic block
print(f"q={q}  shells={len(shells)}  c_block={tuple(int(x) for x in c_block)}")

# Admission record (new function)
admit_shells, admit_weights, dr_rec, sz_rec, final_rank = \
    o12.compute_block_admission_record(shells, c_block, q, gens)

# Reference cascade (untouched function)
sigma, dr_ref, sz_ref, final_ref = o12.compute_block_capacity(shells, c_block, q, gens)

n_shells = len(dr_rec)
assert np.array_equal(dr_rec, dr_ref[:n_shells]), "delta_r mismatch -> different cascade"
assert final_rank == final_ref, f"final_rank mismatch {final_rank} vs {final_ref}"
assert admit_weights.size == final_rank, "admitted count != final_rank"
assert np.all((admit_weights > 0) & (admit_weights <= 1.0 + 1e-9)), "w_j out of (0,1]"

# THE alignment proof: |{m_j > n}| == final_rank - cumsum(delta_r)[n]
ns = np.arange(n_shells)
rank_cum = np.cumsum(dr_rec)
r_pair_from_cumsum = final_rank - rank_cum
r_pair_from_record = np.array([(admit_shells > n).sum() for n in ns])
assert np.array_equal(r_pair_from_record, r_pair_from_cumsum), "support misaligned!"
print("ALIGNMENT OK: |{m_j>n}| == final_rank - cumsum(delta_r) for all n")
print(f"  final_rank={final_rank}  admitted={admit_weights.size}")

# eps_adm via the per-block ansatz
res = adm_entropy_from_admitted(admit_shells, admit_weights, ns)
logr = np.log(np.maximum(r_pair_from_cumsum, 1.0))
print("\n n  r_pair  log r_pair   S_ent   eps_adm   eps/logr")
for n in range(min(n_shells, 9)):
    lr = logr[n]
    eo = res['eps_adm'][n] / lr if lr > 0 else 0.0
    print(f"{n:2d}  {int(r_pair_from_cumsum[n]):5d}   {lr:8.3f}  "
          f"{res['S_ent'][n]:7.3f}  {res['eps_adm'][n]:7.3f}   {eo:6.3f}")

# Sanity on the ansatz outputs
ok_mono = np.all(np.diff(res['S_ent']) <= 1e-9)
ok_eps_nonneg = np.all(res['eps_adm'] >= -1e-12)
mask = logr > 0
ok_sub = np.all(res['eps_adm'][mask] <= logr[mask] + 1e-12)
print(f"\nS_ent monotone non-increasing: {ok_mono}")
print(f"eps_adm >= 0:                  {ok_eps_nonneg}")
print(f"eps_adm <= log r_pair:         {ok_sub}")