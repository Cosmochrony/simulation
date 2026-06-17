"""Chiral transverse frontier diagnostic.

STATUS
------
This script computes the bias-independent existence diagnostic of the chiral transverse
frontier signal. A positive outcome establishes N_A != 0 (existence, not value). It does
NOT by itself prove u != 0, which still requires Schur transport.

It implements the protocol frozen in chiral_transverse_frontier_protocol_spec.md.

DESIGN
------
Two layers, deliberately separated:

  1. Chiral-protocol layer (this file, fully implemented and validated by construction):
     the measured object with the conjugate weight character chi_m*, the gamma5 dressing,
     the eta_m convention phase, the cumulative profiles, and the mandatory controls C1-C5.

  2. Corpus-primitive backend (abstract, MUST be wired to the existing q11_oriented_frontier
     code): group enumeration, admissible generators, BFS metric, A_c / B_c, the metaplectic
     weight characters chi_m, the gamma5 chiral assignment, and the capacity normalisation.
     Reference implementations are given for the paper-defined quantities (A_c, B_c, Delta A_c)
     so they can be checked against the existing code; the genuinely corpus-specific primitives
     raise NotImplementedError until wired, rather than being guessed.

VALIDATION GATE
---------------
Control C5 (raw control reproduces <Delta A_c>_{partial+} = 0 on q in {61,101,151}) is the
independent relation that certifies the group/fingerprint backend is correct. Do not trust any
chiral signal until C5 passes.

INTERPRETATION GATE
-------------------
eta_m must be pinned by local inspection of the Weil-lift convention (chi_m o phi = eta_m chi_{-m})
before the run is interpretable. The script refuses to emit a success verdict while eta_plus is None.

CONVENTIONS
-----------
English code and comments; figures emitted as PDF only; progress/ETA on long runs; milestone
checkpoint/resume so a halted run does not lose computed shells; multicore over shells; all
outputs written to the current working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

from concurrent.futures import ProcessPoolExecutor, as_completed


# ----------------------------------------------------------------------------------------------- #
# Corpus-primitive backend (abstract). Wire these to q11_oriented_frontier.py.                    #
# ----------------------------------------------------------------------------------------------- #


@dataclass
class HeisElement:
    """Element g = (a, b, z) of Heis_3(Z/qZ). Backend may subclass / replace freely."""

    a: int
    b: int
    z: int
    q: int

    def __mul__(self, other: "HeisElement") -> "HeisElement":
        # Heisenberg law: (a,b,z)*(a',b',z') = (a+a', b+b', z+z'+a*b').
        q = self.q
        return HeisElement(
            (self.a + other.a) % q,
            (self.b + other.b) % q,
            (self.z + other.z + self.a * other.b) % q,
            q,
        )

    def inverse(self) -> "HeisElement":
        q = self.q
        return HeisElement((-self.a) % q, (-self.b) % q, (-self.z + self.a * self.b) % q, q)

    def key(self) -> tuple:
        return (self.a, self.b, self.z)


class CorpusBackend:
    """Interface to the corpus-specific definitions.

    Reference implementations are provided for the paper-defined phases A_c, B_c, Delta A_c.
    The genuinely corpus-specific primitives (admissible generators, BFS adjacency, chi_m,
    gamma5 assignment, capacity) are left abstract: implement them by delegating to the
    existing q11_oriented_frontier code. Do NOT guess them here.
    """

    def __init__(self, q: int, central_char: int = 1):
        self.q = q
        self.c = central_char % q  # central character c

    # --- paper-defined: reference implementations (verify against existing code) ------------- #

    def A_c(self, g: HeisElement) -> int:
        """Central phase A_c(g) = c * (gamma - b*a) for the rank-one fingerprint.

        For the single-generator central character used in Q11OF the swept central coordinate
        is z, so A_c(g) = c * z. The (gamma - b*a) form is the multi-index O12 fingerprint;
        reconcile the index convention with the existing fingerprint module.
        """
        return (self.c * g.z) % self.q

    def Delta_A_c(self, g: HeisElement, s: HeisElement) -> int:
        """Directed-edge central increment Delta A_c(g,s) = A_c(g s) - A_c(g).

        For admissible generators with z(s) = 0 this reduces to the pure cocycle c * a(g) * s_b.
        Returned as a signed residue centred in (-q/2, q/2] so averages are unbiased.
        """
        raw = (self.A_c(g * s) - self.A_c(g)) % self.q
        # centre the residue so the symmetric/raw cancellations are exact, not modular artefacts.
        if raw > self.q // 2:
            raw -= self.q
        return raw

    # --- corpus-specific: MUST be wired to q11_oriented_frontier.py --------------------------- #

    def elements(self) -> Iterable[HeisElement]:
        """Enumerate the admissible carrier of Heis_3(Z/qZ) used by the BFS cascade."""
        raise NotImplementedError("Wire CorpusBackend.elements to the q11 group enumeration.")

    def admissible_generators(self) -> Iterable[HeisElement]:
        """Return the admissible generator set s (the Cayley graph generators)."""
        raise NotImplementedError(
            "Wire CorpusBackend.admissible_generators to the q11 admissible generator set."
        )

    def bfs_distance(self, g: HeisElement) -> int:
        """BFS distance d(e, g) in the admissible Cayley graph."""
        raise NotImplementedError("Wire CorpusBackend.bfs_distance to the q11 BFS metric.")

    def chi(self, m: int, g: HeisElement, s: HeisElement) -> complex:
        """Metaplectic weight-m character chi_m(g,s) under W(SO(2)) = U(1)_{J3}.

        The measured object uses chi_m* (the conjugate); take the conjugate at the call site.
        The convention relation chi_m o phi = eta_m chi_{-m} (|eta_m| = 1) is a property of THIS
        function and must be read off here (Lemma 2), not assumed.
        """
        raise NotImplementedError("Wire CorpusBackend.chi to the q11 metaplectic weight character.")

    def gamma5(self, g: HeisElement, s: HeisElement) -> float:
        """Chiral weight +/-1 of the Lorentzian spinorial lift of the edge (g -> g s).

        NOT an attribute of the finite edge alone: it is the chiral weight of the Lorentzian
        spinorial lift. R_b = W(-I) = FT^2 is a scalar on the chiral carrier ([gamma5, R_b] = 0),
        so the dressed signal survives the residual reflection phi while the undressed one re-pairs.
        """
        raise NotImplementedError("Wire CorpusBackend.gamma5 to the q11 Lorentzian chiral lift.")

    def capacity_increment(self, r: int) -> float:
        """Radial/capacity increment Delta I_hat(r) from the B_c sector (AAR radial sector)."""
        raise NotImplementedError("Wire CorpusBackend.capacity_increment to the q11 capacity I_b.")


# ----------------------------------------------------------------------------------------------- #
# Frontier construction.                                                                          #
# ----------------------------------------------------------------------------------------------- #


@dataclass
class Frontier:
    """Directed frontier edges of a BFS shell.

    outgoing = {(g, s) : d(e,g)=r, d(e,gs)=r+1}
    incoming = {(g, s) : d(e,g)=r, d(e,gs)=r-1}
    """

    r: int
    outgoing: list = field(default_factory=list)  # list of (g, s)
    incoming: list = field(default_factory=list)


def build_frontiers(backend: CorpusBackend) -> dict[int, Frontier]:
    """Group directed edges by source-shell radius r."""
    gens = list(backend.admissible_generators())
    frontiers: dict[int, Frontier] = {}
    for g in backend.elements():
        r = backend.bfs_distance(g)
        fr = frontiers.setdefault(r, Frontier(r=r))
        for s in gens:
            gs = g * s
            r_next = backend.bfs_distance(gs)
            if r_next == r + 1:
                fr.outgoing.append((g, s))
            elif r_next == r - 1:
                fr.incoming.append((g, s))
    return frontiers


# ----------------------------------------------------------------------------------------------- #
# Chiral-protocol layer (fully implemented; this is the new content).                             #
# ----------------------------------------------------------------------------------------------- #


def _theta(backend: CorpusBackend, edges: list, m: int, dressed: bool) -> complex:
    """Average of [gamma5] * chi_m* * (2 pi / q) * Delta A_c over a set of directed edges.

    dressed=True  -> includes gamma5 (the physical signal and controls C1, C2, C3).
    dressed=False -> omits gamma5 (control C4, the undressed null).
    """
    if not edges:
        return 0.0 + 0.0j
    q = backend.q
    pref = 2.0 * np.pi / q
    acc = 0.0 + 0.0j
    for g, s in edges:
        weight = np.conjugate(backend.chi(m, g, s))  # chi_m*
        if dressed:
            weight = weight * backend.gamma5(g, s)
        acc += weight * pref * backend.Delta_A_c(g, s)
    return acc / len(edges)


def theta_dressed_outgoing(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Measured object Theta^{partial+}_{gamma5,m}(r)."""
    return _theta(backend, fr.outgoing, m, dressed=True)


def theta_symmetrised(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Control C1: <gamma5 chi_m* Delta A_c>_{partial+ U partial-}; must be ~ 0."""
    return _theta(backend, fr.outgoing + fr.incoming, m, dressed=True)


def theta_reversed_outgoing(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Control C2 on the explicitly paired inverted frontier (partial+ S_r)^{-1}.

    (partial+ S_r)^{-1} = {(g s, s^{-1}) : (g, s) in partial+ S_r}. The reversed edge of an
    outgoing edge of S_r lives in partial- S_{r+1}, NOT partial- S_r, so we build the pairing
    explicitly here rather than indexing partial- at the same r. Expect this = -Theta^{partial+}.
    """
    inv_edges = [(g * s, s.inverse()) for (g, s) in fr.outgoing]
    return _theta(backend, inv_edges, m, dressed=True)


def theta_undressed_outgoing(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Control C4: undressed <chi_m* Delta A_c>_{partial+}; reported, never a physical signal."""
    return _theta(backend, fr.outgoing, m, dressed=False)


def theta_raw_outgoing(backend: CorpusBackend, fr: Frontier) -> complex:
    """Control C5: raw <Delta A_c>_{partial+}; must reproduce 0 (Q11OF). m and chi omitted."""
    if not fr.outgoing:
        return 0.0 + 0.0j
    q = backend.q
    pref = 2.0 * np.pi / q
    return (pref * sum(backend.Delta_A_c(g, s) for (g, s) in fr.outgoing)) / len(fr.outgoing)


# ----------------------------------------------------------------------------------------------- #
# Per-shell evaluation (the unit of parallelism and of checkpointing).                            #
# ----------------------------------------------------------------------------------------------- #


@dataclass
class ShellResult:
    r: int
    n_out: int
    n_in: int
    theta_out: dict  # m -> Theta^{partial+}_{gamma5,m}
    theta_sym: dict  # m -> C1
    theta_rev: dict  # m -> C2 (on inverted frontier)
    theta_undr: dict  # m -> C4
    theta_raw: complex  # C5
    cap_incr: float  # Delta I_hat(r)


def evaluate_shell(backend: CorpusBackend, fr: Frontier, ms=(+1, -1)) -> ShellResult:
    return ShellResult(
        r=fr.r,
        n_out=len(fr.outgoing),
        n_in=len(fr.incoming),
        theta_out={m: theta_dressed_outgoing(backend, fr, m) for m in ms},
        theta_sym={m: theta_symmetrised(backend, fr, m) for m in ms},
        theta_rev={m: theta_reversed_outgoing(backend, fr, m) for m in ms},
        theta_undr={m: theta_undressed_outgoing(backend, fr, m) for m in ms},
        theta_raw=theta_raw_outgoing(backend, fr),
        cap_incr=backend.capacity_increment(fr.r),
    )


# ----------------------------------------------------------------------------------------------- #
# Cumulative profiles and the success criterion.                                                  #
# ----------------------------------------------------------------------------------------------- #


def cumulative_profiles(shells: list[ShellResult], ms=(+1, -1)):
    """Return normalised and unnormalised cumulative profiles Theta^cum_{gamma5,m}(n).

    Normalised: (1 / I_hat(n)) sum_{r<=n} Theta^{partial+}(r) Delta I_hat(r).
    Both forms returned so the normalisation cannot hide a small-depth artefact.
    """
    shells = sorted(shells, key=lambda s: s.r)
    out = {m: {"r": [], "cum_unnorm": [], "cum_norm": []} for m in ms}
    for m in ms:
        acc_w = 0.0 + 0.0j
        acc_unw = 0.0 + 0.0j
        cap_total = 0.0
        for sh in shells:
            acc_w += sh.theta_out[m] * sh.cap_incr
            acc_unw += sh.theta_out[m]
            cap_total += sh.cap_incr
            out[m]["r"].append(sh.r)
            out[m]["cum_unnorm"].append(acc_unw)
            out[m]["cum_norm"].append(acc_w / cap_total if cap_total != 0 else 0.0 + 0.0j)
    return out


def success_verdict(shells, cum, eta_plus: Optional[complex], tol: float):
    """Evaluate C1-C5 and the success criterion.

    Returns a dict of flags. The success verdict is None (undecidable) while eta_plus is None:
    the run is computable but not interpretable until the Lemma 2 convention phase is pinned.
    """
    flags = {}

    # C1: symmetrised cancellation null on every shell.
    flags["C1_symmetrised_null"] = all(
        abs(sh.theta_sym[m]) <= tol for sh in shells for m in sh.theta_sym
    )
    # C2: reversed-frontier sign flip Theta^{(partial+)^{-1}} = -Theta^{partial+}.
    flags["C2_cascade_reversal"] = all(
        abs(sh.theta_rev[m] + sh.theta_out[m]) <= tol for sh in shells for m in sh.theta_out
    )
    # C4: undressed profile re-pairs (does not carry the signal) -> compatible with 0.
    flags["C4_undressed_null"] = all(
        abs(sh.theta_undr[m]) <= tol for sh in shells for m in sh.theta_undr
    )
    # C5: raw control reproduces 0 (the independent backend-correctness gate).
    flags["C5_raw_null"] = all(abs(sh.theta_raw) <= tol for sh in shells)

    # Nonzero transverse signal somewhere.
    flags["transverse_signal_present"] = any(
        abs(v) > tol for v in cum[+1]["cum_norm"]
    )

    # C3: Theta^cum_{+1} = -eta_plus* Theta^cum_{-1}. Undecidable without eta_plus.
    if eta_plus is None:
        flags["C3_weight_relation"] = None
        flags["verdict"] = None  # not interpretable: pin eta_m first.
        return flags

    eta_star = np.conjugate(eta_plus)
    flags["C3_weight_relation"] = all(
        abs(cp + eta_star * cm) <= tol
        for cp, cm in zip(cum[+1]["cum_norm"], cum[-1]["cum_norm"])
    )

    backend_ok = flags["C5_raw_null"] and flags["C1_symmetrised_null"] and flags["C4_undressed_null"]
    flags["verdict"] = bool(
        backend_ok and flags["transverse_signal_present"] and flags["C3_weight_relation"]
    )
    return flags


# ----------------------------------------------------------------------------------------------- #
# Checkpointing / resume.                                                                         #
# ----------------------------------------------------------------------------------------------- #


def _ckpt_path(q: int) -> str:
    return f"./chiral_frontier_ckpt_q{q}.pkl"


def load_checkpoint(q: int) -> dict:
    path = _ckpt_path(q)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return {}


def save_checkpoint(q: int, done: dict) -> None:
    tmp = _ckpt_path(q) + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(done, fh)
    os.replace(tmp, _ckpt_path(q))  # atomic, so a kill mid-write does not corrupt the checkpoint.


# ----------------------------------------------------------------------------------------------- #
# Driver for a single q (with progress/ETA and resume).                                           #
# ----------------------------------------------------------------------------------------------- #


def _worker(args):
    backend_factory, fr = args
    backend = backend_factory()
    return evaluate_shell(backend, fr)


def run_single_q(
    q: int,
    backend_factory: Callable[[], CorpusBackend],
    eta_plus: Optional[complex],
    tol: float = 1e-9,
    workers: Optional[int] = None,
):
    """Compute all shells for one q, resuming from checkpoint, returning the verdict flags."""
    backend = backend_factory()
    frontiers = build_frontiers(backend)
    radii = sorted(frontiers)
    done = load_checkpoint(q)  # r -> ShellResult
    todo = [r for r in radii if r not in done]

    print(f"[q={q}] shells total={len(radii)} done={len(done)} todo={len(todo)}", flush=True)
    t0 = time.time()
    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_worker, (backend_factory, frontiers[r])): r for r in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            done[r] = fut.result()
            save_checkpoint(q, done)  # milestone: persist after every shell.
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - i) / rate if rate > 0 else float("inf")
            print(
                f"[q={q}] shell r={r} ({i}/{len(todo)}) "
                f"elapsed={elapsed:6.1f}s eta={eta:6.1f}s",
                flush=True,
            )

    shells = [done[r] for r in radii]
    cum = cumulative_profiles(shells)
    flags = success_verdict(shells, cum, eta_plus, tol)
    return shells, cum, flags


# ----------------------------------------------------------------------------------------------- #
# Figures (PDF only).                                                                             #
# ----------------------------------------------------------------------------------------------- #


def plot_profiles(q: int, cum: dict) -> None:
    if not _HAVE_MPL:
        print(f"[q={q}] matplotlib unavailable; skipping figure.", flush=True)
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, style in ((+1, "-o"), (-1, "-s")):
        rr = cum[m]["r"]
        ax[0].plot(rr, np.real(cum[m]["cum_norm"]), style, label=f"m={m:+d} (norm)")
        ax[1].plot(rr, np.real(cum[m]["cum_unnorm"]), style, label=f"m={m:+d} (unnorm)")
    for a, title in zip(ax, ("normalised cumulative", "unnormalised cumulative")):
        a.axhline(0.0, color="k", lw=0.6)
        a.set_xlabel("BFS depth n")
        a.set_ylabel(r"Re $\Theta^{\rm cum}_{\gamma_5,m}$")
        a.set_title(f"q={q}: {title}")
        a.legend()
    fig.tight_layout()
    fig.savefig(f"./chiral_frontier_profiles_q{q}.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- #
# Entry point.                                                                                     #
# ----------------------------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--q",
        type=int,
        nargs="+",
        default=[61, 101, 151],
        help="baseline {61,101,151}; extended campaign {61,101,151,211,307} (401 later).",
    )
    ap.add_argument(
        "--eta-plus",
        type=complex,
        default=None,
        help="Lemma 2 convention phase eta_{+1} (|eta|=1). Leave unset until pinned: "
        "the run then computes but stays non-interpretable (verdict=None).",
    )
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.eta_plus is None:
        print(
            "WARNING: eta_plus not pinned. Profiles will be computed but the success verdict "
            "is left undecidable (verdict=None) until the Weil-lift convention is read off.",
            flush=True,
        )

    summary = {}
    for q in args.q:
        # Wire CorpusBackend (or a subclass delegating to q11_oriented_frontier.py) here.
        backend_factory = lambda q=q: CorpusBackend(q)
        shells, cum, flags = run_single_q(
            q, backend_factory, args.eta_plus, tol=args.tol, workers=args.workers
        )
        plot_profiles(q, cum)
        summary[q] = {k: (None if v is None else bool(v)) for k, v in flags.items()}
        print(f"[q={q}] flags: {json.dumps(summary[q])}", flush=True)

    with open("./chiral_frontier_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Wrote ./chiral_frontier_summary.json", flush=True)


if __name__ == "__main__":
    main()