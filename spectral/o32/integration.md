# Unified O12 pipeline, acceleration, and a deterministic [H-color] check

## 1. Unification of the two spectral_O12.py

The o25 (pair) and o32 (triplet) pipelines shipped with slightly different
`spectral_O12.py`. A line-by-line comparison shows the divergence is confined to
two functions, both in the residual-tracking path:

`gram_schmidt_batch_with_residuals` and `compute_block_capacity_with_residuals`.

The o25 version returns normalised residuals `||w_s|| / ||v_s||` and an 8-tuple
(including `basis_mat_window`, `pi_coords`, `shell_all_vecs` for the O26 Test 4 and
Q5a-O5 admissibility weights). The o32-shipped version returns raw residuals
`||w_s||` and a 6-tuple.

Everything that o32 actually imports — `compute_block_capacity`, `build_generators`,
`bfs_shells`, `find_fitting_window` — is byte-identical between the two files. o32
never calls the residual-tracking function. So the sigma observable is computed
identically on both sides, and o32 is agnostic to the divergence.

The unified `spectral_O12.py` therefore adopts the o25 base (the 8-tuple residual
path that o25 depends on) and adds a `normalise=True` flag to
`gram_schmidt_batch_with_residuals`, so the o32-style raw-residual behaviour is
reachable via `normalise=False` without changing the o25 default. Verified: both
pipelines import and run end to end against the unified file (o32 `--quick`
completes; the o25 8-tuple residual path returns its full 16-field result; the
`normalise` flag produces both residual conventions).

If any O26/Q5a script genuinely needs raw residuals, pass `normalise=False`; that
is the only behavioural choice that the two files disagreed on.

## 2. Acceleration (folded into the unified file)

Profiling showed the cost is not the Weil action (`weil_batch_lut` is already an
optimal phase LUT) but `gram_schmidt_batch`: a per-vector Python loop (~10^6
iterations at q=101) with an `np.vstack` that reallocates the whole basis on every
insertion. Since `sigma_c(n) = delta_r_n / |S_n|` is a rank increment, which is
independent of intra-shell processing order, the loop is replaced by, per chunk,
one batched projection plus a `QR(mode='r')` (no tall orthogonal factor) and a
small q-by-q SVD. The Gram-matrix route `W^H W` is tempting but unstable here (the
fingerprints have norm ~1/q, and squaring the conditioning inflates the rank), so
QR + small SVD is used instead.

`compute_block_capacity_fast` is a drop-in for `compute_block_capacity` with the
same signature; verified byte-identical sigma (max difference 0.000e+00) at
q in {29, 61, 101}. Per-block speedup ~2.7x at q=101, ~3x at q=29.

`bfs_shells_depth_capped(gens, q, max_depth)` caps the BFS at a depth instead of a
node fraction. The delta/sigma fit needs only shells 0..n1, and the early shells
are identical across q until the L1 ball wraps the modulus (~depth q/2). At q=401,
depth 15 visits ~2.2e4 nodes instead of ~3.2e6 (149x), with byte-identical sigma
over the fit window. This removes the memory blow-up that made q=401 unreachable
and lets the campaign run on the Mac. Use `max_depth = n1 + buffer` (3 is enough).

Honest magnitude: combined, these give ~2.8x end to end and unblock q=401 on the
memory side, but they do not turn weeks into a day. The dominant campaign cost is
`(#groups) x M x (per-block cost)`, and the per-block cost at q=401 stays heavy
because the rank must reach q. The order-of-magnitude lever for the [H-color]
question is not a larger q but a smaller campaign (Section 3).

## 3. A deterministic, Monte-Carlo-free [H-color] check (`hcolor_exact_check.py`)

O31 (Prop. 4.19) proves the BFS sector rank equality r_c(n)=r_wc(n)=r_w2c(n)
exactly. The open problem (O32 open problem 3) is whether the Born-Infeld
fingerprint observable sigma_c(n) inherits it at finite q. o32 answers this with
Monte-Carlo over random auxiliary (c2,c3), whose sampling variance is exactly what
makes R_var(q) noisy.

The checker removes the sampling. For matched blocks across the colour orbit it
computes sigma exactly and compares it pointwise, shell by shell, with no M and no
averaging. Two matchings are tested:

  A. same auxiliary:    (c,c2,c3) vs (wc,c2,c3) vs (w2c,c2,c3)
  B. scaled auxiliary:  (c,c2,c3) vs (wc,wc2,wc3) vs (w2c,w2c2,w2c3)

Results (q=13 exhaustive over all valid (c2,c3); q=61 over many blocks and two
seeds; q=151 spot check):

  matching A:  max|sigma_i - sigma_j| = O(0.1 to 0.8)   (sigma genuinely varies)
  matching B:  max|sigma_i - sigma_j| = 0.000e+00       (exact to machine precision)

So sigma is exactly pointwise invariant under the orbit action applied to the whole
block character data (c1,c2,c3) -> w*(c1,c2,c3), not under scaling c1 alone. The
mechanism is confirmed: the fingerprint matrices of matched blocks have identical
singular spectra shell by shell (max difference ~1e-15), i.e. they are related by a
fixed unitary on C^q, which preserves all rank increments and hence sigma exactly.

Two consequences:

  - This is precisely the bridge O31 left open ("a direct analytical connection
    between the sector rank equality and the BI-fingerprint observable remains to be
    established"). The numerics show the metaplectic/dilation intertwining acts as a
    genuine unitary on the BI fingerprint, not only isospectrally on the Markov
    operator. The remaining analytical step is finite and concrete: exhibit the
    unitary U_w on C^q realising the Heisenberg dilation automorphism
    (a,b,g) -> (la, lb, l^2 g) with l^2 = w, and show fingerprint_{w.block}(g) =
    U_w fingerprint_{block}(g) for all g. Unitarity then gives [H-color]_pointwise
    exactly. This is far more tractable than a Hecke-spectral Theorem A'.

  - The R_var(q) ~ q^-1 residual in O32 is explained as a finite-M sampling
    artefact, not a genuine finite-q breaking: the o32 worker draws independent
    random (c2,c3) for each character, so it compares E[sigma_c] vs E[sigma_wc]
    over unmatched auxiliaries. The exact symmetry holds for the orbit-matched
    auxiliary, and since w* is a bijection on the auxiliary space it also forces
    E[sigma_c] = E[sigma_wc] exactly (not merely to O(q^-1)). q=401 is therefore not
    needed to settle [H-color].

One refinement worth noting for the paper statement: the exact co-admissibility is
invariance of the block under the orbit action on all three characters, with the
auxiliary characters co-scaling; the c1-only comparison (matching A) is not
invariant. This sharpens how [H-color] should be phrased.

Usage:

  python hcolor_exact_check.py --q 61 --n0 2 --n1 7 --n-orbits 8 --aux-pairs 4
  python hcolor_exact_check.py --q 13 --n0 2 --n1 5 --exhaustive
  python hcolor_exact_check.py --q 151 --n0 3 --n1 12 --n-orbits 2 --aux-pairs 2

## 4. Explicit U_omega, and a closed-form proof of [H-color]_pointwise

Deriving the Weil operator of the dilation makes the unitary explicit and turns the
numerical observation into a proof.

Pure-Fourier-mode structure. In the O12 construction the Weil action is applied to
the uniform seed, so each generator step contributes a phase linear in the index x.
For a block (c1,c2,c3) at a shell element g with accumulated steps
(a_i,b_i,g_i) = g, g*s1, g*s1*s2 (more precisely the three intermediate elements),
the fingerprint is a single additive character:

    fp[x] = q^{-3/2} exp( 2 pi i (A + B x) / q ),
    B = c1 b1 + c2 b2 + c3 b3 (mod q),
    A = c1(g1 - b1 a1) + c2(g2 - b2 a2) + c3(g3 - b3 a3) (mod q).

Verified to machine precision: every fingerprint vector is single-frequency (energy
in one Fourier bin to ~1e-16) at q in {13, 61, 151}.

The dilation operator. Scaling the whole block by omega sends B -> omega B and
A -> omega A. The Weil operator of the dilation automorphism
phi: (a,b,g) -> (la, lb, l^2 g) is the index permutation

    (U_omega f)[x] = f[ omega x mod q ],

a unitary (permutation) on C^q, independent of the block and of g, which sends the
character e_B to e_{omega B}. The fingerprint then transforms as

    fp_{omega . block} = exp( 2 pi i (omega - 1) A / q ) * U_omega fp_{block}.

Verified vector by vector: || fp_{omega.block} - phase * U_omega fp_{block} || ~ 1e-16
at q in {13, 61, 151}. The leading factor is a per-vector unimodular phase, invisible
to Gram-Schmidt.

Proposition ([H-color]_pointwise, exact). Additive characters of Z/qZ are
orthonormal, so a set of fingerprint vectors is linearly independent iff their
frequencies B are distinct. Hence the Gram-Schmidt rank increment of a shell equals
the number of new distinct frequencies it realises, and

    sigma_c(n) = |{ new distinct B at shell n }| / |S_n|.

Scaling the block by omega maps every frequency B -> omega B. Since omega is a unit
mod q, B -> omega B is a bijection of Z/qZ, which preserves the number of distinct
frequencies realised at every shell. Therefore

    sigma_c(n) = sigma_{omega c}(n) = sigma_{omega^2 c}(n)  for all n,

exactly, at finite q, with no Monte-Carlo and no asymptotics. This is the bridge O31
left open, now closed: the metaplectic intertwining acts as a genuine unitary
(the dilation permutation U_omega) on the BI fingerprint, not merely isospectrally on
the Markov operator. Equivalently, sigma is a frequency-counting functional and the
orbit action permutes the frequency set.

Scope and caveat. The proof is for sigma as computed by the O12 pipeline (uniform
seed; the pure-mode structure is exact, verified to 1e-16 and derivable in closed
form). The invariance is under the orbit action on the full block character data
(c1,c2,c3) -> omega(c1,c2,c3); scaling c1 alone is not invariant, since B depends on
all three characters. Because omega* is a bijection on the auxiliary characters, the
o32 averaged statistic E[sigma_c] = E[sigma_{omega c}] also holds exactly, and the
R_var(q) ~ q^-1 residual is a finite-M sampling artefact (independent random
auxiliaries per character), not a finite-q breaking.

## 5. Consequence: the campaign is no longer compute-bound

The pure-mode structure also gives a closed form for sigma that needs no linear
algebra at all: count distinct values of B = c1 b1 + c2 b2 + c3 b3 (mod q) over the
BFS shell and the 4^3 generator triples. `compute_block_capacity_freq` in the unified
spectral_O12.py is a drop-in for `compute_block_capacity` with this method.

Verified sigma-identical (max difference 0.0) at q in {61, 101, 151}, with measured
per-block speedups of 89x (q=61), 296x (q=101), 447x (q=151), growing with q. At
q=307 and q=401, three blocks complete in ~0.6 s, versus the multi-week Gram-Schmidt
path. The premise that q=401 is out of reach no longer holds: the entire O25/O32
campaign at q=401 (and beyond) is now a matter of minutes on the Mac. The delta and
R_var observables can be recomputed across all primes essentially for free.

Files: `verify_Uomega.py` (the vector-by-vector U_omega check), `sigma_freq.py`
(standalone frequency-counting sigma), both also folded into spectral_O12.py.