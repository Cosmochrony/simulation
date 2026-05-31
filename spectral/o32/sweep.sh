cd /Users/javarome/project/@cosmochrony/simulation/spectral/o32
PY=/Users/javarome/project/@cosmochrony/simulation/.venv/bin/python
for q in 61 151 211 307 313 331 337 349 367 373; do
  "$PY" hcolor_exact_check.py --q $q --n0 3 --n1 13 --n-orbits 4 --aux-pairs 3
done