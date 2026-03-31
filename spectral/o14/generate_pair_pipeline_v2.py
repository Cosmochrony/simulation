"""
generate_pair_pipeline_v2.py
=============================
Pipeline paires conjuguees {c, q-c} pour le test de la conjecture O20.

S'appuie directement sur spectral_O12.py (run_one_prime, compute_block_capacity,
bfs_shells, build_generators, sample_generic_blocks).

Pour chaque prime q et chaque paire (c, q-c) choisie :
  - utilise les shells BFS deja calcules (un seul BFS par prime)
  - calcule compute_block_capacity pour un bloc de caractere central c
    et un bloc de caractere central q-c (avec les memes b1, b2)
  - reconstruit Sigma_c(n) et Sigma_{q-c}(n) depuis les increments
  - construit sigma_pair(n) = (Sigma_c(n)/q) * (Sigma_{q-c}(n)/q)
  - detecte les rangs de stabilisation n1 < n2 < n3
  - mesure q^2 * sigma_pair(n3)

Note sur all_sigma du pipeline O12 :
  all_sigma[i, n] = delta_r_n / |S_n|
  (increments de rang normalises par la taille du shell, PAS le span cumule)
  Pour obtenir Sigma_c(n) = sum_{k=0}^{n} delta_r_k
  il faut multiplier par shell_sizes et sommer.

Usage
-----
  python generate_pair_pipeline_v2.py
  python generate_pair_pipeline_v2.py --primes 29 61 --n-pairs 6 --out ./pair_data
"""

import argparse
import os
import sys
import time

import numpy as np

# spectral_O12.py doit etre dans le meme repertoire ou sur PYTHONPATH
try:
    from spectral_O12 import (
        run_one_prime,
        build_generators,
        bfs_shells,
        compute_block_capacity,
        find_fitting_window,
        ols_loglog,
    )
except ImportError:
    print("ERREUR : spectral_O12.py introuvable.")
    print("Placez generate_pair_pipeline_v2.py dans le meme repertoire que spectral_O12.py")
    sys.exit(1)

# =============================================================================
# PARAMETRES
# =============================================================================

PRIMES      = [29, 61, 101, 151, 211]
N_PAIRS     = 6       # paires {c, q-c} par prime
N_MAX_BFS   = {       # profondeur BFS max par prime (O13-compatible)
    29:  8,
    61:  10,
    101: 20,
    151: 37,
    211: 50,
}
BFS_FRAC    = {       # fraction du groupe exploree
    29:  0.99,
    61:  0.99,
    101: 0.99,
    151: 0.29,
    211: 0.106,
}
SEED        = 0
OUT_DIR     = "pair_data"
N_DIM_TGT   = 3       # nombre de dimensions cibles
STAB_FRAC   = 0.15    # fraction du max de delta_Sigma pour detecter une stabilisation

def beta_from_delta(delta):
    return 1.0 / (delta + 0.5)

# =============================================================================
# RECONSTRUCTION DU SPAN CUMULE DEPUIS LES INCREMENTS
# =============================================================================

def reconstruct_cumspan(sigma_diff, shell_sizes):
    """
    sigma_diff[n] = delta_r_n / |S_n|  (sortie de compute_block_capacity)
    shell_sizes[n] = |S_n|
    Retourne Sigma_c = array d'entiers : Sigma_c[n] = sum_{k=0}^{n} delta_r_k
    """
    delta_r = np.round(sigma_diff * shell_sizes).astype(int)
    return np.cumsum(delta_r)

# =============================================================================
# DETECTION DES RANGS DE STABILISATION
# =============================================================================

def detect_stab_ranks(delta_sigma_pair, ns, n_target=3, min_frac=0.15):
    """
    delta_sigma_pair[n] = increment du span pair au rang n
    Detecte les n_target premiers rangs ou l'increment depasse min_frac * max.
    """
    if np.max(delta_sigma_pair) < 1e-12:
        return None
    thr   = min_frac * np.max(delta_sigma_pair)
    peaks = []
    for i, n in enumerate(ns):
        if delta_sigma_pair[i] >= thr:
            if not peaks or int(n) > peaks[-1] + 1:
                peaks.append(int(n))
        if len(peaks) == n_target:
            break
    return peaks if len(peaks) == n_target else None

# =============================================================================
# PIPELINE PAR PRIME
# =============================================================================

def process_prime(q, n_pairs, out_dir, rng):
    print(f"\n{'='*60}")
    print(f"[q={q}]  n_pairs={n_pairs}")

    t0    = time.perf_counter()
    gens  = build_generators(q)
    bfrac = BFS_FRAC.get(q, 0.10)
    nmax  = N_MAX_BFS.get(q, 20)

    # BFS unique pour ce prime
    print(f"  BFS (bfs_frac={bfrac}, n_max={nmax})...")
    shells     = bfs_shells(None, None, gens, q, bfrac)
    n_shells   = len(shells)
    ns         = np.arange(n_shells)
    shell_sizes = np.array([len(s) for s in shells])
    print(f"  {n_shells} shells, {shell_sizes.sum()} noeuds  ({time.perf_counter()-t0:.1f}s)")

    # Choisir n_pairs valeurs c dans {1, ..., q//2}
    c_pool   = np.arange(1, q // 2 + 1)
    n_actual = min(n_pairs, len(c_pool))
    chosen_c = rng.choice(c_pool, size=n_actual, replace=False)

    results  = []

    for c in chosen_c:
        qmc  = q - c

        # Choisir un vecteur graine (b1, b2) aleatoire non nul
        b1, b2 = rng.integers(1, q, size=2)
        c_block_c   = np.array([c,   b1, b2], dtype=np.int64)
        c_block_qmc = np.array([qmc, b1, b2], dtype=np.int64)

        # Capacite pour c et q-c
        sigma_diff_c,   _, sz_c,   _ = compute_block_capacity(
            shells, c_block_c,   q, gens, n_max=nmax)
        sigma_diff_qmc, _, sz_qmc, _ = compute_block_capacity(
            shells, c_block_qmc, q, gens, n_max=nmax)

        # Aligner les longueurs
        n_min = min(len(sigma_diff_c), len(sigma_diff_qmc), len(shell_sizes))
        sigma_diff_c   = sigma_diff_c[:n_min]
        sigma_diff_qmc = sigma_diff_qmc[:n_min]
        sz             = shell_sizes[:n_min]
        ns_loc         = ns[:n_min]

        # Reconstruction du span cumule
        cum_c   = reconstruct_cumspan(sigma_diff_c,   sz)
        cum_qmc = reconstruct_cumspan(sigma_diff_qmc, sz)

        # Observable residuel decroissant : R_c(n) = (q - Sigma_c(n)) / q
        # Analogue direct de all_sigma dans les fichiers O12/O14.
        resid_c   = np.clip(q - cum_c,   0, q).astype(float) / q
        resid_qmc = np.clip(q - cum_qmc, 0, q).astype(float) / q
        sigma_pair = resid_c * resid_qmc

        # Decrements (positifs) pour detection des stabilisations
        delta_cum = -np.diff(resid_c + resid_qmc,
                             prepend=float(resid_c[0]+resid_qmc[0]))
        delta_cum = np.clip(delta_cum, 0, None)

        # Fenetre de fit et delta_pair
        n0, n1 = find_fitting_window(ns_loc[1:], sigma_pair[1:], q)
        n0 = max(n0, 1)
        n1 = min(n1, n_min - 1)
        delta_pair, C_hat, R2 = ols_loglog(ns_loc, sigma_pair, n0, n1)

        # Rangs de stabilisation
        stab = detect_stab_ranks(delta_cum, ns_loc, N_DIM_TGT, STAB_FRAC)

        # sigma_BI et sigma_tilde_BI
        if stab:
            n3     = stab[-1]
            idx_n3 = np.where(ns_loc == n3)[0]
            sBI    = float(sigma_pair[idx_n3[0]]) if len(idx_n3) else np.nan
            stBI   = q**2 * sBI
        else:
            n3, sBI, stBI = None, np.nan, np.nan

        beta_star = beta_from_delta(delta_pair) if not np.isnan(delta_pair) else np.nan

        results.append({
            "q":          q,
            "c":          int(c),
            "qmc":        int(qmc),
            "b1":         int(b1),
            "b2":         int(b2),
            "delta_pair": delta_pair,
            "beta_star":  beta_star,
            "R2":         R2,
            "n0":         n0,
            "n1":         n1,
            "stab":       stab,
            "n3":         n3,
            "sigma_BI":   sBI,
            "stBI":       stBI,
            "ns":         ns_loc,
            "sigma_pair": sigma_pair,
            "cum_c":      cum_c,
            "cum_qmc":    cum_qmc,
        })

        st_s = f"{stBI:.4f}" if not np.isnan(stBI) else "nan"
        print(f"  ({c:3d},{qmc:3d})  delta_pair={delta_pair:.3f}  "
              f"beta*={beta_star:.4f}  R2={R2:.4f}  "
              f"q2*sBI={st_s}  rangs={stab}")

    # Sauvegarder
    if results:
        n_min_save = min(len(r["sigma_pair"]) for r in results)
        out_path   = os.path.join(out_dir, f"q{q}_pairs_v2.npz")
        np.savez(out_path,
            q           = np.int64(q),
            pairs       = np.array([[r["c"], r["qmc"]] for r in results], dtype=np.int64),
            sigma_pair  = np.array([r["sigma_pair"][:n_min_save] for r in results]),
            cum_c       = np.array([r["cum_c"][:n_min_save]      for r in results]),
            cum_qmc     = np.array([r["cum_qmc"][:n_min_save]    for r in results]),
            ns          = results[0]["ns"][:n_min_save].astype(np.int64),
            delta_pair  = np.array([r["delta_pair"] for r in results]),
            beta_star   = np.array([r["beta_star"]  for r in results]),
            r2          = np.array([r["R2"]         for r in results]),
            stBI        = np.array([r["stBI"]        for r in results]),
        )
        print(f"  -> {out_path}")

    print(f"  Temps total q={q} : {time.perf_counter()-t0:.1f}s")
    return results

# =============================================================================
# TABLEAU RECAPITULATIF
# =============================================================================

def print_summary(all_results):
    print("\n" + "="*80)
    print("RECAPITULATIF — q^2 * sigma_pair(n3)  (cible = 9)")
    print("="*80)
    print(f"{'q':>5}  {'(c,q-c)':>10}  {'d_pair':>7}  {'b*':>6}  "
          f"{'q2*sBI':>8}  {'rangs':>14}  {'R2':>6}")
    print("-"*80)
    for r in all_results:
        rs  = str(r["stab"]) if r["stab"] else "—"
        st  = f"{r['stBI']:.4f}" if not np.isnan(r["stBI"]) else "nan"
        print(f"{r['q']:>5}  ({r['c']:>3},{r['qmc']:>3})  "
              f"{r['delta_pair']:>7.3f}  {r['beta_star']:>6.4f}  "
              f"{st:>8}  {rs:>14}  {r['R2']:>6.4f}")
    print("="*80)

    valid = [r for r in all_results if not np.isnan(r["stBI"])]
    if valid:
        stBIs = [r["stBI"] for r in valid]
        print(f"\nq^2*sigma_BI : "
              f"moy={np.mean(stBIs):.4f}  "
              f"std={np.std(stBIs):.4f}  "
              f"median={np.median(stBIs):.4f}  "
              f"(cible=9.0)")
        n9 = sum(1 for x in stBIs if abs(x - 9.0) < 1.5)
        print(f"Paires proches de 9 (+/-1.5) : {n9}/{len(valid)}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primes", nargs="+", type=int, default=PRIMES)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--out",    default=OUT_DIR)
    parser.add_argument("--seed",   type=int, default=SEED)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("="*60)
    print("generate_pair_pipeline_v2.py")
    print(f"Primes : {args.primes}")
    print(f"Paires : {args.n_pairs} par prime")
    print(f"Sortie : {os.path.abspath(args.out)}")
    print("="*60)

    all_results = []
    for q in args.primes:
        all_results.extend(process_prime(q, args.n_pairs, args.out, rng))

    print_summary(all_results)
    print("\nTermine.")

if __name__ == "__main__":
    main()
