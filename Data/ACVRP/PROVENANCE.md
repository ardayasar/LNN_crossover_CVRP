# ACVRP benchmark instances — provenance

Asymmetric CVRP instances from Fischetti, Toth & Vigo (1994), as distributed by
the VRP-REP repository under the dataset name `FTV1994`.

| Field | Value |
|---|---|
| Source | https://www.vrp-rep.org/datasets/item/ftv1994.html |
| Download | https://www.vrp-rep.org/datasets/download/ftv1994.zip |
| Retrieved | 2026-09-22 |
| Instances | 8 |
| Format | TSPLIB-style, `TYPE : ACVRP`, `EDGE_WEIGHT_FORMAT : FULL_MATRIX` |

Reference:

> M. Fischetti, P. Toth, D. Vigo. "A branch-and-bound algorithm for the
> capacitated vehicle routing problem on directed graphs."
> *Operations Research* 42(5), 1994.

The RTF `readme.txt` shipped in the upstream zip is not reproduced here; its
only content is the citation above and the note that the format follows TSPLIB
with distances stored as a full matrix.

## Verification

Distance matrices are genuinely asymmetric (`d != d.T`) for all 8 instances, and
`ceil(sum(demands) / capacity)` equals the vehicle count encoded in each
filename, matching the best-known values hardcoded in `cvrp_loader.py`.

| Instance | Nodes | Capacity | Demand sum | Vehicles | BKS |
|---|---|---|---|---|---|
| A034-02f | 34 | 1000 | 1940 | 2 | 322 |
| A036-03f | 36 | 1000 | 2440 | 3 | 341 |
| A039-03f | 39 | 1000 | 2797 | 3 | 375 |
| A045-03f | 45 | 1000 | 2679 | 3 | 414 |
| A048-03f | 48 | 1000 | 2290 | 3 | 453 |
| A056-03f | 56 | 1000 | 2462 | 3 | 495 |
| A065-03f | 65 | 1000 | 2825 | 3 | 512 |
| A071-03f | 71 | 1000 | 2407 | 3 | 548 |

Note: the BKS values above are those carried in `cvrp_loader.py`. They have not
been re-verified against an independent route checker; treat them as reference
values pending the validation step described in the revision plan.

## Checksums (SHA-256)

```
34d5e612d5c0f01a8f2e32a0887b3c30ebcd73802a2b7ce29c77b830301139bc A034-02f.dat
037efab1c8338de24d64ff7198ef47a2d05aaf4ea0583dbc863b96a63659c0d9 A036-03f.dat
8c1a7b9af294ee19f5a26833d91dc95ef778d3f4e5819e29fbd0b23d965313c8 A039-03f.dat
7176350c3b3be03a64f417d5d38c26c919720c2ce53e81fdd5fcbb73f1415872 A045-03f.dat
1978cae6a086d9b70fdb427c6c6c6a3f62b52365bc79ce64b3798bc915505cda A048-03f.dat
9d7331ed38dc0245aa76460fc6b3be74a1bd04f73e4bb82b4e5d7fccdc18e058 A056-03f.dat
72ec6231c0b42c10ecbaebd90c2937558cf570c8ae19e43557fc1d6c649c8228 A065-03f.dat
0c448647924674481f50248a1ec241cf2bec789f3d7473f6113feb0ca4de7d49 A071-03f.dat
```
