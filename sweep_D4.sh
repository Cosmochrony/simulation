#!/usr/bin/env bash
set -e

echo "=== Appendix D.4 numerical sweeps ==="

# --------------------------------------------------
# 0) Référence (figure officielle du papier)
# --------------------------------------------------
echo
echo ">>> Reference run (softening ON, default parameters)"
python chi_relaxation_validation.py

# --------------------------------------------------
# 1) Robustesse : softening OFF
# --------------------------------------------------
echo
echo ">>> Robustness check: softening OFF"
python chi_relaxation_validation.py --no_softening

# --------------------------------------------------
# 2) Stress test : couplage fort
# --------------------------------------------------
echo
echo ">>> Stress test: softening OFF, K0 = 6.0"
python chi_relaxation_validation.py --no_softening --K0 6.0

# --------------------------------------------------
# 3) Résolution (optionnel, pour usage interne)
# --------------------------------------------------
for N in 24 32 48; do
  echo
  echo ">>> Resolution sweep: N=${N}"
  python chi_relaxation_validation.py --N ${N}
done

echo
echo "=== Sweep completed ==="
