import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

# =============================================
# PARAMÈTRES PHYSIQUES (recalibrés pour cibler m₁=0.5 MeV, m₂=100 MeV, m₃=1 GeV)
# =============================================
L = 32                     # Taille réduite pour tests rapides
K_0 = 1e52                # Couplage ajusté [s⁻²] (pour λ₁ ~ 2.5e54 s⁻²)
a = 1e-15                 # Espacement [m] (échelle nucléaire)
chi_c = 1e-35             # Échelle de χ [m] (échelle de Planck)
c = 2.99792458e8          # Vitesse de la lumière [m/s]
target_masses = [0.5, 100, 1000]  # Masses cibles [MeV]

# =============================================
# 1. CONSTRUCTION DU LAPLACIEN AVEC 3 SOLITONS (topologie tétraédrique)
#    - Modèle la compression critique d'information (intrication comme sur-compression)
# =============================================
def build_laplacian_3d(L, K_0):
    N = L**3
    laplacian = lil_matrix((N, N))

    # Positions des 3 solitons (arrangement tétraédrique pour maximiser la compression)
    soliton_positions = [
        (L//3, L//3, L//3),      # Soliton 1 (électron-like)
        (2*L//3, 2*L//3, L//3),   # Soliton 2 (quark up-like)
        (L//3, 2*L//3, 2*L//3)    # Soliton 3 (quark down-like)
    ]

    for idx in range(N):
        i = idx % L
        j = (idx // L) % L
        k = idx // (L**2)
        neighbors = [
            ((i+1)%L + j*L + k*L**2, -K_0),
            ((i-1)%L + j*L + k*L**2, -K_0),
            (i + ((j+1)%L)*L + k*L**2, -K_0),
            (i + ((j-1)%L)*L + k*L**2, -K_0),
            (i + j*L + ((k+1)%L)*L**2, -K_0),
            (i + j*L + ((k-1)%L)*L**2, -K_0)
        ]

        # Couplage non-uniforme : compression critique près des solitons (modèle l'intrication)
        coupling = K_0
        for (si, sj, sk) in soliton_positions:
            distance = np.sqrt((i-si)**2 + (j-sj)**2 + (k-sk)**2)
            if distance < L//8:  # Région de compression critique
                coupling *= 1e4  # Facteur de compression (1e4 pour modéliser l'intrication)

        laplacian[idx, idx] = 6 * coupling
        for (n_idx, value) in neighbors:
            laplacian[idx, n_idx] = value
    return laplacian.tocsr()

# =============================================
# 2. CALCUL DES VALEURS PROPRES (filtre les modes triviaux)
# =============================================
def compute_eigenvalues(laplacian, n_eigenvalues=5):
    eigenvalues, eigenvectors = eigsh(laplacian, k=n_eigenvalues, which='LM')  # LM pour les plus grandes valeurs propres
    eigenvalues = np.abs(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 1e50]  # Seuil pour éliminer les modes nuls
    return eigenvalues[0:3], eigenvectors[:, 0:3]  # Garde les 3 premiers modes non-triviaux

# =============================================
# 3. CONVERSION EN MASSES (avec ℏ_eff = χ_c²·K_0/c et facteur de compression critique)
# =============================================
def eigenvalues_to_masses(eigenvalues, K_0, a, chi_c, c):
    # Constante de Planck effective : ℏ_eff = χ_c²·K_0 / c (Section 4.10)
    hbar_eff = chi_c**2 * K_0 / c

    # Facteur de conversion : λ_n [s⁻²] → m_n [MeV]
    # m_n = √(λ_n) * (ℏ_eff/c²) * (1/a) * 1e6 (pour MeV)
    # Le facteur 1e-12 ajuste l'échelle de compression critique (intrication)
    conversion_factor = np.sqrt(hbar_eff / (c**2 * 1.602176634e-19)) * (1 / a) * 1e6 * 1e-12
    masses_mev = np.sqrt(eigenvalues) * conversion_factor
    return masses_mev

# =============================================
# EXÉCUTION PRINCIPALE
# =============================================
if __name__ == "__main__":
    print("=== Simulation Cosmochronie : Spectre de masse des 3-solitons ===")
    print(f"Paramètres : L={L}, K_0={K_0:.1e} s⁻², a={a:.1e} m, χ_c={chi_c:.1e} m")

    # 1. Construction du Laplacien avec 3 solitons (modèle la compression critique)
    laplacian = build_laplacian_3d(L, K_0)

    # 2. Calcul des valeurs propres (méthode d'Arnoldi)
    eigenvalues, eigenvectors = compute_eigenvalues(laplacian)
    print(f"Valeurs propres filtrées (s⁻²) : {eigenvalues}")

    # 3. Conversion en masses (utilise ℏ_eff = χ_c²·K_0/c et facteur de compression)
    masses_mev = eigenvalues_to_masses(eigenvalues, K_0, a, chi_c, c)
    print(f"Masses calculées (MeV) : {masses_mev}")
    print(f"Masses cibles (MeV) : {target_masses}")

    # 4. Comparaison avec les cibles (pour l'Appendice D.7)
    print("\n=== RÉSULTATS (pour l'Appendice D.7) ===")
    for n, (computed, target) in enumerate(zip(masses_mev, target_masses)):
        error = abs(computed - target) / target * 100
        print(f"Mode {n+1} : Calculé = {computed:.2f} MeV | Cible = {target:.1f} MeV | Erreur = {error:.1f}%")
