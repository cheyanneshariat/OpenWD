# Metal population consistency

This follow-up to the cool-DZ release (PR #38) makes the level populations
used by opacity consistent with the partition functions that formed each
ion abundance. It retains the released continuum tables, line profiles,
ionization shifts and Rydberg-only neutral occupation prescription.

## Ionization in a helium host

Neutral-He excluded volume applies only to levels with mean radius at least
`NEUTRAL_HARD_SPHERE_MINIMUM_LEVEL_RADIUS_CM`, as in PR #38. Low levels retain
their neutral reservoir; charged occupation probabilities still apply. The
Blouin dense-He ionization shift and the Saha partition ratio are unchanged.
The original PR #37's compensating neutral ground-state ratio was needed by
its all-level excluded-volume prescription. That prescription was superseded
by PR #38, so the compensation is omitted here. Applying it to the released
Rydberg-only partitions would add a spurious ground-state correction.

## Populations used by opacity

The stored partition functions remain absolute sums. A native bound level's
population uses the same statistical weight, Boltzmann factor and occupation
factor as its contribution to that partition. Summing the levels included in
the partition therefore recovers the ion abundance. The native 0.1-eV
partition cutoff is unchanged. Autoionizing resonances retain their separate
historical LTE approximation and are not included in this conservation claim.

Ordinary lines, Verner ground terms, special Mg II/Ca I/Ca II/Na I profiles and the
O I dissolved-series lower populations share this calculation. An optional
profile-dissolution switch cannot disable the occupation factor needed to
normalize the lower population. The trace and bulk state builders currently
use different charged-perturber conventions; each consumer uses the one that
formed its partition rather than silently substituting another density.

## External photoionization levels

TOPbase source indices are not Stout indices. A uniquely identified native
fine-structure level or complete LS term uses the corresponding native level
populations. Each channel keeps its own opening energy; an excited-parent
channel threshold is not the lower state's binding energy.

Some source entries are aggregate model levels without constituent identities.
The default `effective-level` policy retains their supplied excitation and
aggregate statistical weight, with hydrogenic survival calculated from the
true lower binding energy and the existing shell-mean neutral radius. This
is an explicit representative-level approximation. It does not recover the
unknown constituents or guarantee that aggregate populations sum to the
native partition function. The [TLUSTY guide, section
8.1](https://tlusty.oca.eu/tlusty/Tlusty2002/pdf/tlguide200.pdf) distinguishes
such model levels from constituent-resolved populations and describes the
limited accuracy of hydrogenic occupation for complex ions.

`TOPbasePhotoionizationDatabase.unmapped_population_policy="strict"` rejects
unresolved nonideal levels for audits. No channel is silently omitted. Model
and spectrum metadata record the policy, section counts, and its limitation.
Counts describe photoionization sections, not distinct states: several channels
can share a lower state. The ideal-state source-model convention is unchanged.

## Repairs already released in PR #38

The RAP absolute excitation/photon-grid repair and fixed Ca II source
quadrature already ship in PR #38. This follow-up retains them without
reapplying their numerical changes. In particular, RAP absolute photon
support is not clipped at a representative binding. Such aggregate tables
do not define one physical edge, so an explicit dissolved-pseudocontinuum
request is rejected rather than assigned a guessed limit.

The old PR #37 comparison figures predate the cool-DZ release and do not
qualify this combined candidate. They are omitted from the updated change.
Saved-structure comparisons against current main must measure this remaining
population correction separately from the already released RAP/profile work.

## What the tests establish

Ground-only Saha tests evaluate the reference independently. Separate opacity
tests hold ion density fixed, preventing cancellation between ionization and
opacity errors. Further checks cover dilute and disabled-correction limits,
higher ion stages, charge balance, native included-level sums, ground terms,
source-local IDs, excited-parent channels and the pseudocontinuum edge.

These are implementation checks, not validation of all cool-DZ atmospheres or
of inferred abundances. Saved-structure comparisons isolate opacity changes;
new cold-start equilibrium and resolution checks remain distinct requirements.
Frozen spectra and their acceptance tolerances must not be changed merely to
make these source changes pass. Earlier spectra retain their original versions.

## Comparison with the released PR #38 code

With the same saved atmospheres, configurations and wavelength samples:

| Star | Maximum relative flux change | Location | Integrated flux change |
| --- | ---: | ---: | ---: |
| PG1225 | 0.0049231% | 2659.76 A | +0.0002631% |
| J0738 | 0.364745% | 1222 A | +0.0168186% |

Component reversions isolate nearly all of each difference to TOPbase lower
populations. Restoring all previous population expressions recovers every
stored PR #38 flux sample exactly on the capture runtime. The separately
versioned [controls](../../tests/data/approved_regressions/metal_populations_2026_10_24/README.md)
retain all structures and tolerances and include hashes, runtime and component
attribution. Historical spectra continue to run under their original population
convention; the previous references remain unchanged.

All four J1637 fixed spectra pass unchanged. Its public default uses ideal
metal partitions; the new nonideal TOPbase normalization is exercised by the
nonideal material tests and the DZ comparisons. Both DAZ fixed-spectrum controls
also pass. These are fixed-state implementation comparisons, without a fresh
atmosphere or a claim of improved agreement with observations. Full protected
PR qualification is still required.

## Population export

The population state retains its actual `atomic_database` for matching
external photoionization terms during a calculation. This is a runtime
reference, not an extra set of population arrays. Common JSON and NPZ exports
omit fields marked `serialize=False` in dataclass metadata. All physical
population arrays and the existing code/data provenance remain in the output.

Without this distinction, recursively exporting the population atom also
expanded its line-selection caches. Each cached transition refers to its
whole ion, including all that ion's transitions: the resulting output grows
quadratically with the line list. One standard PG1159 selection has 25,532
cached candidates and would expand to 79,474,430 transition records per
exported atom. This occurs after the expensive solve, explaining why the
failed CI runs reached the same final logged residuals as the successful
predecessor before their runners were shut down.

The regression tests forbid walking the runtime cache, check that both PG1159
material states and their physical arrays still export without pickle, and
verify that output size is independent of cache size. A diagnostic with the
real PG1159 atom and a four-layer Planck state exported 2,877 arrays to a
1.08-MB archive in about 0.08 seconds. This checks export behavior, not a new
atmosphere-convergence certificate. No atmosphere, spectrum, tolerance or
iteration limit is changed by this export repair.
