for s in 120 160 200 260 320 400; do
  python toy_cosmochrony_1d_a.py --N 2000 --steps 8000 --lr 0.02 --sep $s \
    --pin --kappa_pin 0.05 --k_eigs 12 --no_plots
done
