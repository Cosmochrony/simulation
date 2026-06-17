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

  1. Primary layer (this file, fully implemented and validated by construction): the UNWEIGHTED
     chiral-dressed measured object <gamma5 (2pi/q) Delta A_c>_{partial+} -- the direct frontier
     observable of the Q14 operator coefficient alpha -- and its controls C_raw, C_sym, C_rev. No
     chi_m weight enters the primary criterion.

  2. Auxiliary chi_m representation diagnostics (reported, not the N_A criterion): the weighted
     dressed/undressed objects with expected phi-relations of OPPOSITE sign
     (dressed Theta_m = +eta_m* Theta_{-m}; undressed E_m = -eta_m* E_{-m}). The opposite signs are
     the structural fingerprint of the gamma5 dressing. The weighted-dressed object is NOT the split
     carrier: its split projection Theta_{+1} - Theta_{-1} vanishes for eta=1.

  3. Corpus-primitive backend (abstract, MUST be wired to the existing q11_oriented_frontier code):
     group enumeration, admissible generators, BFS metric, A_c / B_c, gamma5 chiral assignment, the
     capacity normalisation, and (for the auxiliary layer only) the metaplectic weight characters
     chi_m. Reference implementations are given for A_c, B_c, Delta A_c; the genuinely
     corpus-specific primitives raise NotImplementedError until wired, rather than being guessed.

VALIDATION GATE
---------------
Control C_raw (raw control reproduces <Delta A_c>_{partial+} = 0 on q in {61,101,151}) is the
independent relation that certifies the group/fingerprint backend is correct. Do not trust any
chiral signal until C_raw passes.

INTERPRETATION GATE
-------------------
The PRIMARY verdict is decidable WITHOUT eta_m: the measured object is unweighted. The new primary
lock is gamma5 o phi = -gamma5 (Q11OF) and the consistency <gamma5 Delta A_c>_{partial+} <-> alpha
(Q14 sec 6.3). eta_m is required only for the auxiliary chi_m relation checks, which are reported
separately and left undecidable until the Weil-lift convention chi_m o phi = eta_m chi_{-m} is read
off.

CONVENTIONS
-----------
English code and comments; figures emitted as PDF only; progress/ETA on long runs; milestone
checkpoint/resume so a halted run does not lose computed shells; multicore over shells; all
outputs written to the current working directory.
"""

from __future__ import annotations

import argparse
import functools
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


def _theta_weighted(backend: CorpusBackend, edges: list, m: int, dressed: bool) -> complex:
    """Average of [gamma5] * chi_m* * (2 pi / q) * Delta A_c. AUXILIARY (chi_m representation).

    dressed=True  -> Theta_m^dr, expected relation Theta_m = +eta_m* Theta_{-m}.
    dressed=False -> E_m^undr,   expected relation E_m    = -eta_m* E_{-m}.
    These are a representation diagnostic only, NOT the N_A criterion (see module docstring).
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


def _theta_unweighted(backend: CorpusBackend, edges: list, dressed: bool) -> complex:
    """Average of [gamma5] * (2 pi / q) * Delta A_c, NO chi_m weight. PRIMARY object.

    dressed=True  -> the measured signal <gamma5 Delta A_c> (the operator coefficient alpha).
    dressed=False -> the raw control <Delta A_c> (phi-killed; backend-correctness gate).
    """
    if not edges:
        return 0.0 + 0.0j
    q = backend.q
    pref = 2.0 * np.pi / q
    acc = 0.0 + 0.0j
    for g, s in edges:
        w = backend.gamma5(g, s) if dressed else 1.0
        acc += w * pref * backend.Delta_A_c(g, s)
    return acc / len(edges)


def _inverted(fr: Frontier) -> list:
    """Explicitly paired inverted frontier (partial+ S_r)^{-1} = {(g s, s^{-1})}."""
    return [(g * s, s.inverse()) for (g, s) in fr.outgoing]


# --- primary (unweighted) signal and controls ------------------------------------------------- #


def theta_primary(backend: CorpusBackend, fr: Frontier) -> complex:
    """Primary measured object Theta^{partial+}_{gamma5}(r) = <gamma5 (2pi/q) Delta A_c>."""
    return _theta_unweighted(backend, fr.outgoing, dressed=True)


def control_raw(backend: CorpusBackend, fr: Frontier) -> complex:
    """C_raw: <(2pi/q) Delta A_c>_{partial+}; must reproduce 0 (Q11OF). Backend-correctness gate."""
    return _theta_unweighted(backend, fr.outgoing, dressed=False)


def control_sym(backend: CorpusBackend, fr: Frontier) -> complex:
    """C_sym: <gamma5 (2pi/q) Delta A_c>_{partial+ U (partial+)^{-1}}; anti-bias, must be ~ 0."""
    return _theta_unweighted(backend, fr.outgoing + _inverted(fr), dressed=True)


def control_reversed(backend: CorpusBackend, fr: Frontier) -> complex:
    """C_rev: signal on the inverted frontier; expect = -theta_primary (alpha_rev = -alpha)."""
    return _theta_unweighted(backend, _inverted(fr), dressed=True)


# --- auxiliary (chi_m representation) diagnostics --------------------------------------------- #


def aux_dressed(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Auxiliary Theta_m^dr; expected Theta_m = +eta_m* Theta_{-m}. Not the N_A criterion."""
    return _theta_weighted(backend, fr.outgoing, m, dressed=True)


def aux_undressed(backend: CorpusBackend, fr: Frontier, m: int) -> complex:
    """Auxiliary E_m^undr; expected E_m = -eta_m* E_{-m}. Not the N_A criterion."""
    return _theta_weighted(backend, fr.outgoing, m, dressed=False)


# ----------------------------------------------------------------------------------------------- #
# Per-shell evaluation (the unit of parallelism and of checkpointing).                            #
# ----------------------------------------------------------------------------------------------- #


@dataclass
class ShellResult:
    r: int
    n_out: int
    n_in: int
    # primary (unweighted)
    theta_primary: complex  # <gamma5 (2pi/q) Delta A_c>_{partial+}
    c_raw: complex  # <(2pi/q) Delta A_c>_{partial+}, must be 0
    c_sym: complex  # symmetrised on partial+ U (partial+)^{-1}, must be 0
    c_rev: complex  # on (partial+)^{-1}, expect = -theta_primary
    cap_incr: float  # Delta I_hat(r)
    # auxiliary (chi_m representation), reported only
    aux_dr: dict  # m -> Theta_m^dr,  expected Theta_m = +eta_m* Theta_{-m}
    aux_undr: dict  # m -> E_m^undr,    expected E_m    = -eta_m* E_{-m}


def evaluate_shell(backend: CorpusBackend, fr: Frontier, ms=(+1, -1)) -> ShellResult:
    return ShellResult(
        r=fr.r,
        n_out=len(fr.outgoing),
        n_in=len(fr.incoming),
        theta_primary=theta_primary(backend, fr),
        c_raw=control_raw(backend, fr),
        c_sym=control_sym(backend, fr),
        c_rev=control_reversed(backend, fr),
        cap_incr=backend.capacity_increment(fr.r),
        aux_dr={m: aux_dressed(backend, fr, m) for m in ms},
        aux_undr={m: aux_undressed(backend, fr, m) for m in ms},
    )


# ----------------------------------------------------------------------------------------------- #
# Cumulative profiles and the success criterion.                                                  #
# ----------------------------------------------------------------------------------------------- #


def cumulative_profiles(shells: list[ShellResult], ms=(+1, -1)):
    """Cumulative profiles.

    Primary: Theta^cum_{gamma5}(n) = (1/I_hat(n)) sum_{r<=n} theta_primary(r) Delta I_hat(r),
    plus the unnormalised sum. Auxiliary: per-m dressed/undressed cumulatives (representation only).
    """
    shells = sorted(shells, key=lambda s: s.r)
    primary = {"r": [], "cum_unnorm": [], "cum_norm": []}
    aux_dr = {m: {"r": [], "cum_norm": []} for m in ms}
    aux_undr = {m: {"r": [], "cum_norm": []} for m in ms}

    acc_w = 0.0 + 0.0j
    acc_unw = 0.0 + 0.0j
    cap_total = 0.0
    acc_dr = {m: 0.0 + 0.0j for m in ms}
    acc_undr = {m: 0.0 + 0.0j for m in ms}
    for sh in shells:
        acc_w += sh.theta_primary * sh.cap_incr
        acc_unw += sh.theta_primary
        cap_total += sh.cap_incr
        primary["r"].append(sh.r)
        primary["cum_unnorm"].append(acc_unw)
        primary["cum_norm"].append(acc_w / cap_total if cap_total != 0 else 0.0 + 0.0j)
        for m in ms:
            acc_dr[m] += sh.aux_dr[m] * sh.cap_incr
            acc_undr[m] += sh.aux_undr[m] * sh.cap_incr
            denom = cap_total if cap_total != 0 else 1.0
            aux_dr[m]["r"].append(sh.r)
            aux_dr[m]["cum_norm"].append(acc_dr[m] / denom)
            aux_undr[m]["r"].append(sh.r)
            aux_undr[m]["cum_norm"].append(acc_undr[m] / denom)
    return {"primary": primary, "aux_dr": aux_dr, "aux_undr": aux_undr}


def success_verdict(shells, cum, eta_plus: Optional[complex], tol: float):
    """Evaluate the primary controls and verdict, plus the auxiliary chi_m relation checks.

    The PRIMARY verdict is independent of eta_plus: the measured object is unweighted. The
    auxiliary chi_m relations require eta_plus and are reported separately (None until pinned).
    """
    flags = {}

    # --- primary controls (unweighted) ------------------------------------------------------- #
    # C_raw: <Delta A_c>_{partial+} = 0 (backend-correctness gate).
    flags["C_raw_null"] = all(abs(sh.c_raw) <= tol for sh in shells)
    # C_sym: symmetrised dressed cancellation null.
    flags["C_sym_null"] = all(abs(sh.c_sym) <= tol for sh in shells)
    # C_rev: signal on inverted frontier = -theta_primary (alpha_rev = -alpha).
    flags["C_rev_signflip"] = all(
        abs(sh.c_rev + sh.theta_primary) <= tol for sh in shells
    )
    # Nonzero primary signal somewhere (shells, normalised, or unnormalised cumulative).
    flags["signal_present"] = (
        any(abs(sh.theta_primary) > tol for sh in shells)
        or any(abs(v) > tol for v in cum["primary"]["cum_norm"])
        or any(abs(v) > tol for v in cum["primary"]["cum_unnorm"])
    )

    backend_ok = flags["C_raw_null"] and flags["C_sym_null"]
    flags["verdict"] = bool(backend_ok and flags["signal_present"] and flags["C_rev_signflip"])

    # --- auxiliary chi_m relation checks (representation only; require eta_plus) -------------- #
    if eta_plus is None:
        flags["aux_dressed_relation"] = None  # expected Theta_m = +eta* Theta_{-m}
        flags["aux_undressed_relation"] = None  # expected E_m    = -eta* E_{-m}
        return flags

    eta_star = np.conjugate(eta_plus)
    flags["aux_dressed_relation"] = all(
        abs(cp - eta_star * cm) <= tol  # dressed: PLUS sign -> cp = +eta* cm
        for cp, cm in zip(cum["aux_dr"][+1]["cum_norm"], cum["aux_dr"][-1]["cum_norm"])
    )
    flags["aux_undressed_relation"] = all(
        abs(cp + eta_star * cm) <= tol  # undressed: MINUS sign -> cp = -eta* cm
        for cp, cm in zip(cum["aux_undr"][+1]["cum_norm"], cum["aux_undr"][-1]["cum_norm"])
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
    primary = cum["primary"]
    rr = primary["r"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(rr, np.real(primary["cum_norm"]), "-o", label="primary (norm)")
    ax[1].plot(rr, np.real(primary["cum_unnorm"]), "-o", label="primary (unnorm)")
    # auxiliary chi_m dressed cumulatives, for representation context only
    for m, style in ((+1, "--^"), (-1, "--v")):
        ax[0].plot(
            cum["aux_dr"][m]["r"],
            np.real(cum["aux_dr"][m]["cum_norm"]),
            style,
            alpha=0.5,
            label=f"aux m={m:+d} (norm)",
        )
    for a, title in zip(ax, ("normalised cumulative", "unnormalised cumulative")):
        a.axhline(0.0, color="k", lw=0.6)
        a.set_xlabel("BFS depth n")
        a.set_ylabel(r"Re $\Theta^{\rm cum}_{\gamma_5}$")
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
        help="Lemma 2 convention phase eta_{+1} (|eta|=1) for the AUXILIARY chi_m checks only. "
        "The primary unweighted verdict does not need it; leave unset until the convention is pinned.",
    )
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    if args.eta_plus is not None and abs(abs(args.eta_plus) - 1.0) > 1e-9:
        ap.error("--eta-plus must satisfy |eta| = 1.")

    if args.eta_plus is None:
        print(
            "NOTE: eta_plus not pinned. The PRIMARY verdict (unweighted <gamma5 Delta A_c>) is still "
            "decidable. Only the auxiliary chi_m relation checks are left undecidable (None) until "
            "the Weil-lift convention chi_m o phi = eta_m chi_{-m} is read off (Lemma 2).",
            flush=True,
        )

    summary = {}
    for q in args.q:
        # Wire CorpusBackend (or a subclass delegating to q11_oriented_frontier.py) here.
        # functools.partial is picklable across ProcessPoolExecutor; a lambda is not.
        backend_factory = functools.partial(CorpusBackend, q)
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