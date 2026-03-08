import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class CosmoSim:
    def __init__(self, name="Q8", c_chi=1.0):
        self.name = name
        self.c_chi = float(c_chi)  # Substrate flux bound
        self._setup_graph()

    def _setup_graph(self):
        if self.name == "Q8":
            # Q8 adjacency matrix (6-regular)
            # Vertex order: 1, -1, i, -i, j, -j, k, -k
            self.A = np.array([
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0]
            ], dtype=float)
            self.N = 8
            self.d = 6
        else:
            raise ValueError(f"Unsupported graph name: {self.name}")

        self.L = self.d * np.eye(self.N) - self.A
        self.vals, self.vecs = np.linalg.eigh(self.L)

        # Sector identification for Q8
        # rho4 (non-abelian): lambda = 6 (indices 1, 2, 3, 4)
        # rho_k (abelian): lambda = 8 (indices 5, 6, 7)
        self.idx_na = np.where(np.isclose(self.vals, 6.0))[0]
        self.idx_ab = np.where(np.isclose(self.vals, 8.0))[0]

        if len(self.idx_na) == 0 or len(self.idx_ab) == 0:
            raise RuntimeError("Could not identify the non-abelian or abelian sectors.")

    def dbi_dynamics(self, t, y):
        chi = y[:self.N]
        v = y[self.N:]

        # Laplacian restoring force
        force = -self.L @ chi

        # DBI freezing factor: (1 - v^2 / c^2)^(3/2)
        # Numerical clipping avoids NaNs near the saturation limit
        v_ratio = np.clip(v**2 / self.c_chi**2, 0.0, 0.999)
        freeze_factor = (1.0 - v_ratio)**1.5

        accel = force * freeze_factor
        return np.concatenate([v, accel])

    def run(self, init_scale=0.5, duration=50.0, num_points=1000):
        # Initial condition: mixture of near-saturating modes
        chi0 = np.zeros(self.N)
        v0 = np.zeros(self.N)

        # Excite one mode from each sector with the same raw amplitude
        v0 += init_scale * self.vecs[:, self.idx_na[0]]
        v0 += init_scale * self.vecs[:, self.idx_ab[0]]

        t_eval = np.linspace(0.0, duration, num_points)
        sol = solve_ivp(
            self.dbi_dynamics,
            [0.0, duration],
            np.concatenate([chi0, v0]),
            t_eval=t_eval,
            method="RK45"
        )
        return sol

    def sector_energies(self, sol):
        v_t = sol.y[self.N:, :].T

        # Kinetic-sector proxy:
        # E_sector = sum_n (v . psi_n)^2 over the selected sector
        e_na = np.sum((v_t @ self.vecs[:, self.idx_na])**2, axis=1)
        e_ab = np.sum((v_t @ self.vecs[:, self.idx_ab])**2, axis=1)
        return e_na, e_ab

    def plot_results(self, sol, init_scale, output_dir="figures", show=False):
        e_na, e_ab = self.sector_energies(sol)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        scale_tag = str(init_scale).replace(".", "p")
        pdf_file = output_path / f"q8_dbi_multimode_scale_{scale_tag}.pdf"

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            sol.t,
            e_na,
            label=f"Non-Abelian Sector ($\\lambda={self.vals[self.idx_na[0]]:.1f}$)",
            lw=2.5
        )
        ax.plot(
            sol.t,
            e_ab,
            label=f"Abelian Sector ($\\lambda={self.vals[self.idx_ab[0]]:.1f}$)",
            lw=1.5,
            alpha=0.8
        )

        ax.set_title(
            f"Multi-Modal DBI Dynamics on {self.name}: Cascade Hypothesis "
            f"(init_scale = {init_scale})"
        )
        ax.set_xlabel("Time")
        ax.set_ylabel("Projected sector energy")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

        fig.tight_layout()
        fig.savefig(pdf_file, format="pdf", bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)
        return pdf_file


if __name__ == "__main__":
    sim = CosmoSim("Q8")

    for init_scale in (0.7, 0.8, 0.9):
        solution = sim.run(init_scale=init_scale, duration=50.0, num_points=1000)
        pdf_path = sim.plot_results(solution, init_scale=init_scale, output_dir="figures")
        print(f"Saved: {pdf_path}")