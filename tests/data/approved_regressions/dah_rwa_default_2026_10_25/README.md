# DAH public default with RWA and magnetic EOS, 2026-10-25

From this date the DAH public default adds stationary-state RWA bound-free
opacity (Rohrmann 2026) with the magnetic Saha/HM EOS and centered-motion
populations above 1 MG, on the unchanged nonmagnetic DA structure. These
fixed-atmosphere spectra are the reviewed outputs of that default on the eight
frozen paper atmospheres in `tests/data/dah_paper`. The paper spectra
themselves are untouched and remain the controls of the named paper
prescription (`paper_dah_config`).

The capture first recovers each frozen paper spectrum with the paper preset
(largest difference 1.1e-13, tolerance 2e-8), then records the default. Largest
changes from the paper spectra: 5.8-12.3% near 3800 A below 80 MG visible field,
27-45% for J1254, J1018 and J1351, and 18.6% at 4254 A for J2247. Integrated
flux changes range from -5.1% (J2247) to +6.3% (J1351). J1351 and J2247 exceed
200 MG and carry the RWA validity domain note.

Capture is offline via `research/capture_dah_rwa_default_controls.py`; tests
never run it. The manifest includes input/output hashes, source hashes, request
identities and measured differences. The capture relaxes no atmosphere and
provides no cold-start certificate. No tolerance was changed.
