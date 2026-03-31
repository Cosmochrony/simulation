"""
spectral_o20_final.py
======================
Test de la conjecture O20 : q^2 * sigma_pair(n3) = 9

Lit les fichiers q<q>_pairs_v2.npz produits par generate_pair_pipeline_v2.py.

Format attendu de q<q>_pairs_v2.npz :
    q           : int
    pairs       : (n_pairs, 2)              valeurs (c, q-c)
    sigma_pair  : (n_pairs, n_depths)       sigma_c * sigma_{q-c}  (decroissant)
    cum_c       : (n_pairs, n_depths)       Sigma_c(n) cumule
    cum_qmc     : (n_pairs, n_depths)       Sigma_{q-c}(n) cumule
    ns          : (n_depths,)               rangs BFS
    delta_pair  : (n_pairs,)                exposant fitte
    beta_star   : (n_pairs,)
    r2          : (n_pairs,)
    stBI        : (n_pairs,)                q^2 * sigma_pair(n3)

Usage
-----
    python spectral_o20_final.py /chemin/vers/pair_data_final
    python spectral_o20_final.py /chemin/vers/pair_data_final --primes 151 211
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PRIMES   = [29, 61, 101, 151, 211]
CIBLE    = 9.0

def beta_from_delta(d):
    return 1.0 / (d + 0.5)

# =============================================================================
# CHARGEMENT
# =============================================================================

def load(q, data_dir):
    path = os.path.join(data_dir, f"q{q}_pairs_v2.npz")
    if not os.path.exists(path):
        print(f"  [q={q}] ABSENT : {path}")
        return None
    data = np.load(path, allow_pickle=True)
    print(f"  [q={q}] Charge : {path}")
    return data

# =============================================================================
# DETECTION DES RANGS DE STABILISATION (sur sigma_pair decroissant)
# =============================================================================

def detect_stab_ranks(sp, ns, n_target=3, frac=0.15):
    drop = -np.diff(sp.astype(float), prepend=sp[0])
    drop = np.clip(drop, 0, None)
    if np.max(drop) < 1e-15:
        return None
    thr   = frac * np.max(drop)
    peaks = []
    for i, n in enumerate(ns):
        if drop[i] >= thr:
            if not peaks or int(n) > peaks[-1] + 1:
                peaks.append(int(n))
        if len(peaks) == n_target:
            break
    return peaks if len(peaks) == n_target else None

# =============================================================================
# PIPELINE PAR PRIME
# =============================================================================

def process_prime(q, data_dir):
    data = load(q, data_dir)
    if data is None:
        return []

    pairs      = data["pairs"]        # (n_pairs, 2)
    sigma_pair = data["sigma_pair"]   # (n_pairs, n_depths) — decroissant
    ns         = data["ns"].astype(int)
    dp_arr     = data["delta_pair"]
    r2_arr     = data["r2"]
    stBI_arr   = data["stBI"]         # q^2 * sigma_pair(n3) deja calcule

    results = []
    for i in range(len(pairs)):
        c, qmc = int(pairs[i, 0]), int(pairs[i, 1])
        sp     = sigma_pair[i]
        dp     = float(dp_arr[i])
        R2     = float(r2_arr[i])
        stBI   = float(stBI_arr[i])

        # Recalcul des rangs de stabilisation (pour coherence)
        stab = detect_stab_ranks(sp, ns)
        n3   = stab[-1] if stab else None

        # Si stBI manquant ou nan dans le npz, recalculer
        if np.isnan(stBI) and stab:
            idx = np.where(ns == n3)[0]
            stBI = q**2 * float(sp[idx[0]]) if len(idx) else np.nan

        beta = beta_from_delta(dp) if not np.isnan(dp) else np.nan

        results.append({
            "q": q, "c": c, "qmc": qmc,
            "delta_pair": dp, "beta": beta, "R2": R2,
            "stab": stab, "n3": n3,
            "stBI": stBI,
            "sp": sp, "ns": ns,
        })

        st_s = f"{stBI:.4f}" if not np.isnan(stBI) else "nan"
        print(f"    ({c:4d},{qmc:4d})  dp={dp:.3f}  b*={beta:.4f}  "
              f"R2={R2:.4f}  q2*sBI={st_s}  rangs={stab}")

    return results

# =============================================================================
# TABLEAU ET STATISTIQUES
# =============================================================================

def print_table(all_results):
    print("\n" + "="*88)
    print(f"TABLEAU — conjecture q^2*sigma_pair(n3) = {CIBLE}")
    print("="*88)
    print(f"{'q':>5}  {'(c,qmc)':>11}  {'dp':>7}  {'b*':>6}  "
          f"{'q2*sBI':>9}  {'rangs':>14}  {'R2':>6}")
    print("-"*88)
    for r in all_results:
        rs  = str(r["stab"]) if r["stab"] else "—"
        st  = f"{r['stBI']:.4f}" if not np.isnan(r["stBI"]) else "nan"
        print(f"{r['q']:>5}  ({r['c']:>4},{r['qmc']:>4})  "
              f"{r['delta_pair']:>7.3f}  {r['beta']:>6.4f}  "
              f"{st:>9}  {rs:>14}  {r['R2']:>6.4f}")
    print("="*88)

    print(f"\n=== STATISTIQUES PAR PRIME ===")
    for q in sorted(set(r["q"] for r in all_results)):
        sub  = [r for r in all_results if r["q"] == q]
        dps  = [r["delta_pair"] for r in sub if not np.isnan(r["delta_pair"])]
        stBIs = [r["stBI"] for r in sub if not np.isnan(r["stBI"])]
        in_w = sum(1 for d in dps if 7.4 <= d <= 10.6)
        if stBIs:
            print(f"  q={q:3d}: {len(sub)} paires  "
                  f"dp={np.mean(dps):.3f}±{np.std(dps):.3f}  "
                  f"in[7.4,10.6]={in_w}/{len(dps)}  "
                  f"q2*sBI={np.mean(stBIs):.3f}±{np.std(stBIs):.3f}  "
                  f"(cible={CIBLE})")

    # Verdict global
    all_stBI = [r["stBI"] for r in all_results
                if not np.isnan(r["stBI"]) and 7.4 <= r["delta_pair"] <= 10.6]
    if all_stBI:
        m, s = np.mean(all_stBI), np.std(all_stBI)
        n9   = sum(1 for x in all_stBI if abs(x - CIBLE) < 2.0)
        print(f"\n  Paires dans [7.4,10.6] : {len(all_stBI)}")
        print(f"  q2*sBI global : {m:.3f} ± {s:.3f}  (cible={CIBLE})")
        print(f"  Proches de 9 (±2) : {n9}/{len(all_stBI)}")
        ecart = abs(m - CIBLE)
        if ecart < max(s, 0.5):
            print(f"  VERDICT : COHERENT (ecart={ecart:.3f} < max(std,0.5)={max(s,0.5):.3f})")
        else:
            print(f"  VERDICT : TENSION  (ecart={ecart:.3f} > max(std,0.5)={max(s,0.5):.3f})")

# =============================================================================
# FIGURES
# =============================================================================

def make_figures(all_results):
    if not all_results:
        return

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        r"O20 — Test conjecture $q^2\,\sigma_{\mathrm{BI}} = 9$"
        "  (vraies paires conjuguees)",
        fontsize=13
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    qs   = np.array([r["q"]          for r in all_results])
    dp   = np.array([r["delta_pair"] for r in all_results])
    st   = np.array([r["stBI"]       for r in all_results])
    bs   = np.array([r["beta"]       for r in all_results])

    mask_valid  = ~np.isnan(st)
    mask_window = mask_valid & (dp >= 7.4) & (dp <= 10.6)

    # (A) q^2*sBI vs delta_pair — panneau principal
    ax = fig.add_subplot(gs[0, 0])
    if mask_valid.sum() > 0:
        sc = ax.scatter(dp[mask_valid], st[mask_valid],
                        c=qs[mask_valid], cmap="viridis",
                        s=60, alpha=0.85, zorder=3)
        plt.colorbar(sc, ax=ax, label="$q$")
    ax.axhline(CIBLE, color="firebrick", lw=1.8, ls="--", label=f"cible={CIBLE}")
    ax.axvspan(7.4, 10.6, alpha=0.10, color="seagreen", label="[7.4,10.6]")
    ax.set_xlabel(r"$\delta_{\mathrm{pair}}$")
    ax.set_ylabel(r"$q^2\,\sigma_{\mathrm{BI}}$")
    ax.set_title(r"(A) $\widetilde{\sigma}_{\mathrm{BI}}$ vs $\delta_{\mathrm{pair}}$")
    ax.legend(fontsize=8)

    # (B) q^2*sBI par prime — paires dans [7.4,10.6] seulement
    ax = fig.add_subplot(gs[0, 1])
    if mask_window.sum() > 0:
        sc2 = ax.scatter(qs[mask_window], st[mask_window],
                         c=qs[mask_window], cmap="viridis",
                         s=60, alpha=0.85, zorder=3)
        plt.colorbar(sc2, ax=ax, label="$q$")
    ax.axhline(CIBLE, color="firebrick", lw=1.8, ls="--", label=f"cible={CIBLE}")
    ax.set_xlabel("Prime $q$")
    ax.set_ylabel(r"$q^2\,\sigma_{\mathrm{BI}}$")
    ax.set_title(r"(B) Paires dans $[7.4,10.6]$ seulement")
    ax.legend(fontsize=8)

    # (C) histogramme q^2*sBI (paires dans [7.4,10.6])
    ax = fig.add_subplot(gs[0, 2])
    vals = st[mask_window]
    if len(vals) > 0:
        ax.hist(vals, bins=max(5, len(vals)//2),
                color="steelblue", alpha=0.7, edgecolor="white")
        ax.axvline(CIBLE, color="firebrick", lw=1.8, ls="--",
                   label=f"cible={CIBLE}")
        ax.axvline(np.mean(vals), color="steelblue", lw=1.2, ls=":",
                   label=f"moy={np.mean(vals):.2f}")
    ax.set_xlabel(r"$q^2\,\sigma_{\mathrm{BI}}$")
    ax.set_ylabel("paires")
    ax.set_title(r"(C) Distribution (paires dans [7.4,10.6])")
    ax.legend(fontsize=8)

    # (D) profils sigma_pair pour chaque prime
    ax = fig.add_subplot(gs[1, 0])
    cmap = plt.cm.viridis
    q_vals = sorted(set(r["q"] for r in all_results))
    colors = {q: cmap(i / max(len(q_vals)-1, 1))
              for i, q in enumerate(q_vals)}
    for r in all_results:
        sp = r["sp"].astype(float)
        ns_r = r["ns"]
        valid = (ns_r >= 1) & (sp > 1e-15)
        if valid.sum() < 2:
            continue
        ax.loglog(ns_r[valid]+1, sp[valid],
                  '-', color=colors[r["q"]], lw=0.8, alpha=0.6)
        if r["stab"]:
            n3  = r["stab"][-1]
            idx = np.where(ns_r == n3)[0]
            if len(idx):
                ax.plot(n3+1, sp[idx[0]], 'o',
                        color=colors[r["q"]], ms=5, zorder=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], color=colors[q], lw=1.5, label=f"q={q}")
               for q in q_vals]
    ax.legend(handles=handles, fontsize=7)
    ax.set_xlabel("$n+1$")
    ax.set_ylabel(r"$\sigma_{\mathrm{pair}}(n)$")
    ax.set_title("(D) Profils sigma_pair (• = n3)")
    ax.grid(True, alpha=0.3, which="both")

    # (E) convergence de q^2*sBI vers 9 en fonction de q
    ax = fig.add_subplot(gs[1, 1])
    for q in q_vals:
        sub = [r for r in all_results
               if r["q"] == q and not np.isnan(r["stBI"])
               and 7.4 <= r["delta_pair"] <= 10.6]
        if sub:
            vals_q = [r["stBI"] for r in sub]
            ax.errorbar(q, np.mean(vals_q), yerr=np.std(vals_q),
                        fmt='o', color=colors[q], ms=7,
                        capsize=4, lw=1.5)
    ax.axhline(CIBLE, color="firebrick", lw=1.8, ls="--",
               label=f"cible={CIBLE}")
    ax.set_xlabel("Prime $q$")
    ax.set_ylabel(r"$q^2\,\sigma_{\mathrm{BI}}$ (moy ± std)")
    ax.set_title("(E) Convergence vers 9 avec q")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (F) relation O7
    ax = fig.add_subplot(gs[1, 2])
    dr = np.linspace(3, 15, 300)
    ax.plot(dr, [beta_from_delta(d) for d in dr], "k-", lw=1.5, label="O7")
    if mask_valid.sum() > 0:
        sc3 = ax.scatter(dp[mask_valid], bs[mask_valid],
                         c=qs[mask_valid], cmap="viridis",
                         s=40, alpha=0.75, zorder=3)
    ax.axvspan(7.4, 10.6, alpha=0.10, color="seagreen")
    ax.axhspan(0.09, 0.13, alpha=0.10, color="steelblue",
               label=r"$\beta^*\in[0.09,0.13]$")
    ax.set_xlabel(r"$\delta_{\mathrm{pair}}$")
    ax.set_ylabel(r"$\beta^*$")
    ax.set_title("(F) Relation O7")
    ax.legend(fontsize=7)
    ax.set_xlim(3, 15)
    ax.set_ylim(0.04, 0.30)

    for ext in ("pdf", "png"):
        fig.savefig(f"spectral_o20_final.{ext}", bbox_inches="tight", dpi=150)
    print("\nFigures : spectral_o20_final.pdf / .png")
    plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default=".",
                        help="Repertoire contenant les q<q>_pairs_v2.npz")
    parser.add_argument("--primes", nargs="+", type=int, default=PRIMES)
    args = parser.parse_args()

    print("="*60)
    print("spectral_o20_final.py")
    print(f"Repertoire : {os.path.abspath(args.data_dir)}")
    print(f"Primes     : {args.primes}")
    print("="*60)

    all_results = []
    for q in args.primes:
        all_results.extend(process_prime(q, args.data_dir))

    if not all_results:
        print("Aucun resultat — verifier que les fichiers "
              "q<q>_pairs_v2.npz sont presents.")
        return

    print_table(all_results)
    make_figures(all_results)
    print("\nTermine.")

if __name__ == "__main__":
    main()
