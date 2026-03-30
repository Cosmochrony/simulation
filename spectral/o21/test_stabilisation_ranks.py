"""
test_stabilisation_ranks.py
============================
Test numérique de la conjecture sur les 3 rangs de stabilisation.

Ce script est AUTONOME et ne requiert pas les fichiers npz du pipeline O12/O13.
Il recrée la dynamique de Gram-Schmidt sur les représentations de Weil
pour des petits primes (q ≤ 61), où la fenêtre BFS est complètement accessible.

OBJECTIF :
  Pour chaque paire (c, q-c) :
    1. Calculer Sigma_c(n) et Sigma_{q-c}(n) séparément (blocs indépendants)
    2. Calculer sigma_pair(n) = (Sigma_c/q) * (Sigma_{q-c}/q)  [définition O16]
    3. Identifier les rangs n_1, n_2, n_3 où Sigma_c + Sigma_{q-c} atteint 1, 2, 3
    4. Calculer sigma_BI_emp = sigma_pair(n_3)
    5. Vérifier la stabilité de sigma_BI_emp entre paires et entre primes

NOTE sur le régime :
  Les exposants δ_pair ≈ 7.44 des papiers O12-O16 sont extraits sur des fenêtres
  de 5-13 shells dans le régime pré-saturation, pour des primes q ∈ {29..211}.
  Ce script travaille dans le régime plus court mais complet : tous les shells
  jusqu'à saturation. Il ne recalcule pas δ_pair mais teste la conjecture
  structurelle sur les rangs de stabilisation et σ_BI.

Paramètres
----------
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES
# ============================================================
PRIMES     = [11, 13, 17, 19, 23, 29]  # petits primes pour accès complet
N_PAIRS    = 4                          # paires par prime
DIM_TARGET = 3                          # dimensions stables attendues
EPS_GS     = 1e-10                      # seuil Gram-Schmidt
# β* de référence (depuis O16-O19 : δ_pair ≈ 7.44)
BETA_STAR_REF = 1.0 / (7.44 + 0.5)

# ============================================================
# GROUPE DE HEISENBERG Heis_3(Z/qZ)
# ============================================================

def heis_mult(u, v, q):
    a, b, t   = u
    a2, b2, t2 = v
    return ((a + a2) % q, (b + b2) % q, (t + t2 + a * b2) % q)

def heis_generators(q):
    return [(1,0,0), (q-1,0,0), (0,1,0), (0,q-1,0)]

# ============================================================
# REPRÉSENTATION DE WEIL ρ_c
# [ρ_c(a,b,t) ψ](x) = ω^{c(bx+t)} ψ(x-a)
# ============================================================

def weil_action(psi, a, b, t, c, q):
    omega = np.exp(2j * np.pi * c / q)
    xs = np.arange(q)
    phases = omega ** (b * xs + t)
    return phases * psi[(xs - a) % q]

# ============================================================
# BFS COMPLET
# ============================================================

def bfs_shells_full(q):
    identity = (0, 0, 0)
    generators = heis_generators(q)
    visited = {identity}
    frontier = [identity]
    yield 0, list(frontier)
    while frontier:
        new_frontier = []
        for u in frontier:
            for g in generators:
                v = heis_mult(u, g, q)
                if v not in visited:
                    visited.add(v)
                    new_frontier.append(v)
        if not new_frontier:
            break
        frontier = new_frontier
        yield len(visited) - len(frontier), frontier  # shell index approx

# ============================================================
# SPAN TRACKER POUR UN BLOC
# ============================================================

def span_tracker(q, c):
    """
    Retourne Sigma_c(n) pour n = 0, 1, ..., jusqu'à saturation à q.
    """
    v0 = np.zeros(q, dtype=complex)
    v0[0] = 1.0
    basis = []
    Sigma = 0
    Sigma_list = []
    shell_idx = 0

    for _, shell_elements in bfs_shells_full(q):
        delta = 0
        for (a, b, t) in shell_elements:
            w = weil_action(v0, a, b, t, c, q)
            r = w.copy()
            for e in basis:
                r -= np.dot(e.conj(), r) * e
            nrm = np.linalg.norm(r)
            if nrm > EPS_GS:
                basis.append(r / nrm)
                delta += 1
        Sigma += delta
        Sigma_list.append(Sigma)
        shell_idx += 1
        if Sigma >= q:
            break

    return np.array(Sigma_list)

# ============================================================
# ANALYSE D'UNE PAIRE
# ============================================================

def analyse_pair(q, c):
    """
    Calcule les quantités clés pour la paire (c, q-c).
    """
    c2 = (q - c) % q

    Sigma_c  = span_tracker(q, c)
    Sigma_qc = span_tracker(q, c2)

    # Aligner les longueurs
    n_min = min(len(Sigma_c), len(Sigma_qc))
    Sigma_c  = Sigma_c[:n_min]
    Sigma_qc = Sigma_qc[:n_min]

    # sigma_pair(n) = (Sigma_c/q) * (Sigma_qc/q)  [définition O16]
    sigma_c   = Sigma_c  / float(q)
    sigma_qc  = Sigma_qc / float(q)
    sigma_pair = sigma_c * sigma_qc

    # Sigma_sum = Sigma_c + Sigma_qc : dimensions cumulées de la paire
    Sigma_sum = Sigma_c + Sigma_qc

    # Rangs de stabilisation : première fois que Sigma_sum >= d pour d=1,2,3
    ranks = []
    for d in range(1, DIM_TARGET + 1):
        idx = np.where(Sigma_sum >= d)[0]
        ranks.append(int(idx[0]) if len(idx) > 0 else None)
    n1, n2, n3 = ranks

    # sigma_BI empirique : valeur de sigma_pair au rang n3
    if n3 is not None and n3 < len(sigma_pair):
        sigma_BI_emp = float(sigma_pair[n3])
    else:
        sigma_BI_emp = np.nan

    return {
        'q': q, 'c': c, 'c2': c2,
        'Sigma_c':  Sigma_c,
        'Sigma_qc': Sigma_qc,
        'sigma_pair': sigma_pair,
        'n1': n1, 'n2': n2, 'n3': n3,
        'sigma_BI_emp': sigma_BI_emp,
    }

# ============================================================
# SÉLECTION DES PAIRES
# ============================================================

def canonical_pairs(q, n_pairs):
    pairs = []
    for c in range(1, q):
        c2 = (q - c) % q
        if c < c2:
            pairs.append((c, c2))
        if len(pairs) >= n_pairs:
            break
    return pairs

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def run(primes=PRIMES, n_pairs=N_PAIRS):
    all_results = []

    for q in primes:
        pairs = canonical_pairs(q, n_pairs)
        for (c, c2) in pairs:
            res = analyse_pair(q, c)
            all_results.append(res)

    return all_results

# ============================================================
# AFFICHAGE
# ============================================================

def print_table(results):
    header = (f"{'q':>5} {'c':>4} {'q-c':>4} "
              f"{'n1':>4} {'n2':>4} {'n3':>4} "
              f"{'σ_BI_emp':>12}  note")
    sep = "=" * len(header)
    print("\n" + sep)
    print("RANGS DE STABILISATION ET σ_BI EMPIRIQUE")
    print(sep)
    print(header)
    print("-" * len(header))

    sigma_BIs = []
    for r in results:
        note = ""
        if r['n3'] is None:
            note = "dim 3 non atteinte"
        elif r['n1'] == r['n2'] == r['n3']:
            note = "dégénéré (3 dim au même rang)"
        elif r['n1'] < r['n2'] < r['n3']:
            note = "3 rangs distincts ✓"
        else:
            note = f"partiellement distincts"

        print(f"{r['q']:>5} {r['c']:>4} {r['c2']:>4} "
              f"{str(r['n1']):>4} {str(r['n2']):>4} {str(r['n3']):>4} "
              f"{r['sigma_BI_emp']:>12.6f}  {note}")

        if not np.isnan(r['sigma_BI_emp']):
            sigma_BIs.append(r['sigma_BI_emp'])

    print(sep)

    if sigma_BIs:
        arr = np.array(sigma_BIs)
        print(f"\nσ_BI empirique — statistiques sur {len(arr)} paires :")
        print(f"  moyenne  = {arr.mean():.6f}")
        print(f"  std      = {arr.std():.6f}")
        print(f"  min/max  = {arr.min():.6f} / {arr.max():.6f}")
        print(f"  CV (std/mean) = {arr.std()/arr.mean():.4f}  "
              f"(< 0.1 = invariant structurel, > 0.3 = dépendant de q)")
        print(f"\nConclusion préliminaire :")
        if arr.std() / arr.mean() < 0.15:
            print("  σ_BI est STABLE entre paires et primes → cohérent avec un invariant structurel.")
        else:
            print("  σ_BI varie significativement → dépend de q ou de la paire.")
            print("  Interprétation : σ_BI n'est pas un invariant universel à ce stade.")

    print(f"\nRéférence β* = {BETA_STAR_REF:.4f} (depuis δ_pair ≈ 7.44, O16-O19)")

# ============================================================
# FIGURES
# ============================================================

def make_figures(results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Rangs de stabilisation et $\\sigma_{\\mathrm{BI}}$ empirique\n"
        "Test conjecture : 3 dimensions stables dans la fenêtre de cascade",
        fontsize=12
    )
    ax1, ax2, ax3, ax4 = axes.flat

    # Panel A : sigma_pair(n) pour toutes les paires
    for r in results:
        sp = r['sigma_pair']
        ns = np.arange(len(sp))
        lbl = f"q={r['q']},({r['c']},{r['c2']})"
        ax1.semilogy(ns + 1, np.maximum(sp, 1e-10), lw=1, alpha=0.7, label=lbl)
        # marquer n3
        n3 = r['n3']
        if n3 is not None and n3 < len(sp) and sp[n3] > 1e-10:
            ax1.scatter([n3 + 1], [sp[n3]], s=30, zorder=5, color='red')
    ax1.set_xlabel('rang BFS $n+1$')
    ax1.set_ylabel(r'$\sigma_{\mathrm{pair}}(n)$ (log)')
    ax1.set_title('Panel A — profils $\\sigma_{\\mathrm{pair}}(n)$\n(point rouge = $n_3$)')
    ax1.legend(fontsize=5, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel B : σ_BI empirique par paire
    labels = [f"q={r['q']}\n({r['c']},{r['c2']})" for r in results]
    sigma_BIs = [r['sigma_BI_emp'] for r in results]
    valid = [(l, s) for l, s in zip(labels, sigma_BIs) if not np.isnan(s)]
    if valid:
        lv, sv = zip(*valid)
        ax2.bar(range(len(lv)), sv, color='steelblue', alpha=0.8)
        mean_val = np.mean(sv)
        ax2.axhline(mean_val, color='red', lw=1.5, linestyle='--',
                    label=f'moyenne = {mean_val:.4f}')
        ax2.set_xticks(range(len(lv)))
        ax2.set_xticklabels(lv, fontsize=6)
        ax2.set_ylabel(r'$\sigma_{\mathrm{BI}}^{\mathrm{emp}} = \sigma_{\mathrm{pair}}(n_3)$')
        ax2.set_title('Panel B — $\\sigma_{\\mathrm{BI}}$ empirique par paire\n'
                      '(invariance attendue)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

    # Panel C : rangs n1, n2, n3 vs q
    qs_seen = sorted(set(r['q'] for r in results))
    for d, marker, color in zip([1,2,3], ['o','s','^'], ['tab:blue','tab:orange','tab:green']):
        r_vals = []
        q_vals = []
        for r in results:
            nk = [r['n1'], r['n2'], r['n3']][d-1]
            if nk is not None:
                q_vals.append(r['q'])
                r_vals.append(nk)
        if q_vals:
            ax3.scatter(q_vals, r_vals, marker=marker, s=40,
                        color=color, label=f'$n_{d}$', alpha=0.7)
    ax3.set_xlabel('prime $q$')
    ax3.set_ylabel('rang de stabilisation $n_k$')
    ax3.set_title('Panel C — rangs $n_1, n_2, n_3$ vs $q$\n'
                  '(rangs distincts = 3 générations séparées)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel D : histogramme de sigma_BI
    sv_all = [r['sigma_BI_emp'] for r in results if not np.isnan(r['sigma_BI_emp'])]
    if sv_all:
        ax4.hist(sv_all, bins=15, color='steelblue', alpha=0.8, edgecolor='white')
        ax4.axvline(np.mean(sv_all), color='red', lw=2, linestyle='--',
                    label=f'moy = {np.mean(sv_all):.4f}')
        ax4.axvline(np.mean(sv_all) + np.std(sv_all), color='gray',
                    lw=1, linestyle=':', label=f'±1σ')
        ax4.axvline(np.mean(sv_all) - np.std(sv_all), color='gray', lw=1, linestyle=':')
        ax4.set_xlabel(r'$\sigma_{\mathrm{BI}}^{\mathrm{emp}}$')
        ax4.set_ylabel('count')
        ax4.set_title('Panel D — distribution de $\\sigma_{\\mathrm{BI}}$\n'
                      '(pic étroit = invariant structurel)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('test_stabilisation_ranks.pdf', bbox_inches='tight')
    plt.savefig('test_stabilisation_ranks.png', dpi=150, bbox_inches='tight')
    print("\nFigures : test_stabilisation_ranks.pdf / .png")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Localisation des rangs de stabilisation et σ_BI empirique")
    print(f"Primes : {PRIMES}")
    print(f"Paires par prime : {N_PAIRS}")
    print(f"Dimension cible : {DIM_TARGET}")
    print()

    results = run()
    print_table(results)
    make_figures(results)
