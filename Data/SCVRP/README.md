# Symmetric CVRP instances

123 instances with matching reference solutions. Layout is flat because
`cvrp_loader.load_all_instances()` does not recurse (see
[the integration guide](../../README.md#-status--known-issues), §3.2).

| Family | Count | Customers | Role | Distance convention |
|---|---|---|---|---|
| **X** (Uchoa et al. 2017) | 100 | 100–1000 | primary held-out test | nearest-integer Euclidean |
| **Golden** 9–20 | 12 | 240–483 | secondary family test | **unrounded** Euclidean |
| **E** | 8 | 21–100 | development / continuity | nearest-integer Euclidean |
| **M** | 2 | 150, 199 | supplementary | nearest-integer Euclidean |
| **F** | 1 | 134 | supplementary | **unresolved** — see below |

Provenance, checksums and per-instance metadata are in
[`INSTANCE_REGISTRY.csv`](INSTANCE_REGISTRY.csv): source URL, retrieval date,
SHA-256 of both files, node/customer counts, capacity, fleet policy, distance
policy, published and recomputed reference cost, reference route count, and
whether the reference exceeds the filename's `k`.

Retrieved 2026-09-23 from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/instances/1).

## Verification performed

Every one of the 123 reference solutions was checked independently of the GA's
decoder: each customer served exactly once, every route within capacity, and
the cost recomputed from coordinates. **All 123 pass.**

The distance convention was then *determined empirically* per instance by
recomputing under both conventions and comparing to the declared cost, rather
than trusting the `EDGE_WEIGHT_TYPE` header:

- **Set X — nearest-integer, 100/100.** Unambiguous.
- **Golden — unrounded, 12/12.** Every Golden file declares `EUC_2D`, yet all
  12 reference costs match only the unrounded computation (e.g. Golden_9:
  declared 579.7020, unrounded 579.7021, rounded 484). The header does not
  determine the convention.
- **F-n135-k7 — matches neither.** Declared 1162; nearest-integer gives 1159
  (off by 3), unrounded gives 1170.65 (off by 8.65). Do not use this instance
  for a reference gap until the convention is identified.

## Two things that will break naive use

**1. The loader cannot see these files.** `load_all_instances()` iterates a
hardcoded list of eight `E-n*` names (`cvrp_loader.py`); it does not scan the
directory. Adding files here changes nothing until the loader takes a manifest
or globs the folder.

**2. Golden needs an unrounded distance path, which does not exist.** The
loader's only distance computation is `round(np.hypot(...))`. Loading Golden
today would silently produce costs ~20% below the reference (Golden_9: 484 vs
579.70) and every reported gap would be meaningless.

## Fleet semantics

**22 of the 100 Set-X reference solutions use more routes than the `k` in their
filename**, by up to 6 routes — e.g. X-n101-k25 uses 26, X-n336-k84 uses 86.
Treating `k` as a hard vehicle cap would reject these published optima. Set X
is recorded as `fleet_policy = unrestricted` in the registry; the other families
are marked `unverified` because their historical protocols differ.
