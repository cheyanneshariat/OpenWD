# J1637 independent-Kurucz reference

This version uses the qualified default atom. The warm-v1 and warm-v2-Fe-II
archives are unchanged and still exercised with their original atomic policies.
All profile/metric and fixed-spectrum tolerances remain unchanged.

The seed is copied exactly from the checksum-verified warm-v2 capsule: fixed
Teff, log g, composition, column mass, optical-depth grid, base temperature and
prescribed 1.5% perturbation. Its cold certificate belongs to the archived source;
these two measured updates do not establish current-code cold convergence.
The manifest pins the seed/output, original source, qualified catalogue,
code/data fingerprints and measured differences from warm-v2.

The fixed spectrum uses the base temperature, not either perturbed update.
At R=2500 with the baseline continuum transfer frozen, the squared residual
improves 0.34% against the bundled observation and 0.81% against the paper model.
Native line pixels change by up to 37.6%, with −0.0873% integrated flux change
over 3400–7700 Å. No physical parameter, abundance or candidate continuum is fitted.

Explicit offline reproduction (choose a new output directory):

```sh
PYTHONPATH=src python tools/build_d6_regression.py \
  tests/data/d6_regressions/warm_v2_feii \
  results/j1637-new-reference \
  --from-frozen-reference --spectrum-first
```

The capture ran with one numerical thread under a 4 GiB RSS watchdog; peak
RSS was 3.973 GiB. Tests never invoke the reference builder or update archives.
