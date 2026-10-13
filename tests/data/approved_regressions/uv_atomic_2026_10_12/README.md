# UV atomic fixed-atmosphere controls (2026-10-12)

Missing Fe II ultraviolet transitions and neutral trace-H metal-line widths. Historical structures, configurations, grids and regression tolerances are unchanged. Disabling only these two changes under the scoped legacy helper recovers the previous DZ microfield-host and DAZ Si-Kurucz controls before capture. Earlier controls remain intact. The DAZ grids are optical; their Fe II partition inputs include the single added upper level.

The manifest records source/input/output hashes, numerical versions, configuration, spectrum diagnostics and the legacy-control proof. These spectra are not a cold-start or atmosphere-equilibrium qualification. Capture uses research/capture_uv_atomic_controls.py and refuses an existing directory.
