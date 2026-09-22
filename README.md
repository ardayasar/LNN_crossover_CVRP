# LNN Crossover for the Capacitated Vehicle Routing Problem (CVRP)

**Repository**: `AlparslanGuzey/LNN_crossover_CVRP`
**DOI**: [10.5281/zenodo.15762345](https://doi.org/10.5281/zenodo.15762345)

An experimental framework for benchmarking classical and learned crossover
operators — including a Liquid Neural Network (LNN) crossover — on the
Capacitated Vehicle Routing Problem. It supports both permutation and
random-key encodings, with and without mutation, and emits per-run CSV logs for
downstream statistical analysis and supervised LNN training.

> **Status:** research code under revision. `GA.py` runs end to end, but the
> committed results in `results/` were produced by an earlier version of the
> pipeline and several methodological defects remain open. See
> [Status & known issues](#-status--known-issues) before relying on any output.

---

## 🚀 Features

- **Six benchmarked crossover operators**
  - Permutation-encoded: `OX`, `CX`, `AEX`, `LNN`
  - Random-key-encoded: `BRKGA`, `RK-2PX`

  Three more are implemented but **not wired into the benchmark**: `PMX`
  (`pmx_crossover`), `RK-U` (`rk_uniform`), and `RK-GR` (`rk_greedy`). The active
  set is hardcoded as `ALLOWED_OPS` inside `run_instance_all_operators`
  (`GA.py:360`).

- **Mutation regimes** — every operator is run twice, with and without mutation:
  - `mut_perm` — swap-2 (permutation encoding)
  - `mut_rk` — bounded uniform jitter, `±0.1` clamped to `[0,1]` (random-key encoding)

- **Local search** — 2-opt, applied to the **permutation encoding only**
  (`GA.py:290`). Note this is *giant-tour* 2-opt, not route-level: it minimizes
  the TSP length of the whole permutation, which is not the objective `fitness`
  measures. See known issue 8.

- **Split decoding** — optimal giant-tour → routes dynamic program (`split_cost`),
  used as the fitness evaluator for both encodings.

- **CSV logging**
  - Per-run metrics → `results/<instance>_summary_<mut|nomut>.csv`
  - Per-generation best cost → `results/convergence/<instance>/<op>/<regime>/run_<n>_convergence.csv`
  - Optional LNN training triples → `GA_LNN/data/` (enable with `LNN_LOG=1`)

- **LNN crossover** — a liquid-inspired recurrent scorer (PyTorch) that maps
  per-customer features to scores, decoded by descending argsort into a child
  permutation. See [Model](#-model).

---

## ⚙️ Installation

### Quick start (recommended)

```bash
git clone https://github.com/AlparslanGuzey/LNN_crossover_CVRP.git
cd LNN_crossover_CVRP
./setup.sh
source .venv/bin/activate
```

`setup.sh` creates a virtualenv, installs the **correct PyTorch build for the
current host**, installs the remaining dependencies, and verifies the result.

| Host | PyTorch build selected |
|---|---|
| macOS / arm64 | default PyPI wheel (CPU + MPS) |
| Linux / aarch64 + CUDA | `cu130` — Grace-Blackwell, e.g. DGX Spark (GB10, `sm_121`) |
| Linux / x86_64 + CUDA | `cu128` |
| anything else | CPU-only |

Options:

```bash
./setup.sh --cpu                      # force the CPU-only build
./setup.sh --venv /path/to/env        # custom virtualenv location
TORCH_CUDA_CHANNEL=cu128 ./setup.sh   # override the CUDA wheel channel
PYTHON=python3.12 ./setup.sh          # pin the interpreter
./setup.sh --help
```

**Requires Python ≥ 3.10.** The script searches for `python3.13` → `python3.12`
→ `python3.11` → `python3.10` → `python3` and uses the first that qualifies.
On macOS, the system `/usr/bin/python3` (3.9) is too old for current PyTorch;
install a newer one with `brew install python@3.12`.

### Manual install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # CPU/MPS build of torch
```

On a CUDA host, install `torch` from the matching wheel index *first*, then the
rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu130 torch
pip install -r requirements.txt
```

### Data

Place CVRPLIB instance files under:

```
Data/SCVRP/      # symmetric,  .vrp + .sol   (8 CVRPLIB E-n* instances)
Data/ACVRP/      # asymmetric, .dat          (8 Fischetti-Toth-Vigo instances)
```

Both sets are included in the repository. The asymmetric set comes from the
[VRP-REP FTV1994 dataset](https://www.vrp-rep.org/datasets/item/ftv1994.html);
source, retrieval date, checksums and consistency checks are recorded in
[`Data/ACVRP/PROVENANCE.md`](Data/ACVRP/PROVENANCE.md).

**Depot indexing.** `GA.py` assumes node index 0 is the depot. Benchmark files
do not all agree: CVRPLIB `E-n*` names node 1 (already index 0), while the
FTV1994 instances name node *N*, the last node. The loader reads
`DEPOT_SECTION` and permutes demands and **both axes** of the distance matrix
so the depot lands at index 0 (`cvrp_loader.py:42`). It raises if the declared
depot has nonzero demand or falls outside the node range.

Best-known costs are parsed from the `Cost` line of the matching `.sol` file
(`cvrp_loader.py:139`). Instance discovery and path logic live in
`cvrp_loader.py:154`.

---

## 🧪 Usage

Run the benchmark:

```bash
python GA.py
```

Generate LNN training data while running:

```bash
LNN_LOG=1 python GA.py
```

Train the LNN crossover:

```bash
python -m GA_LNN.train_lnn \
    --log_dir GA_LNN/data \
    --epochs 200 \
    --batch 128 \
    --lr 3e-4 \
    --wd 1e-4 \
    --alpha 0.2 \
    --aug
```

### Configuration

Module-level constants in `GA.py:28`:

```python
FEATURE_DIM     = 3      # rank-in-parent-1 • rank-in-parent-2 • normalized demand
POP_SIZE        = 25
MAX_GENERATIONS = 400
CROSSOVER_RATE  = 1.0
MUTATION_RATE   = 0.10
NUM_RUNS        = 15
```

Two selections are **not** module-level constants and must be edited in place:

- **Instances** — `TARGET` in the `__main__` block (`GA.py:397`). Defaults to
  `{"E-n22-k4"}`, i.e. a single instance; every other instance is skipped.
- **Operators** — `ALLOWED_OPS` inside `run_instance_all_operators` (`GA.py:360`).

### Progress reporting

Both the benchmark and training print live progress to **stderr**, so piped
stdout (result tables) stays clean:

```bash
python GA.py                 # progress on the terminal
python GA.py > results.log   # tables to the file, progress still on screen
python GA.py 2> progress.log # the reverse
PROGRESS=0 python GA.py      # silence it
```

On a terminal it rewrites one line with a bar, count, elapsed and ETA. When
redirected to a file it writes a timestamped line every 30 s instead, so logs
stay readable. Long-running sweeps emit a heartbeat every 5 s even when no task
has finished, showing how many runs are still in flight.

`smoke_test.py` is the fast way to check a change before committing to a full
sweep - it exercises the whole pipeline in about a second.

### Running concurrent benchmark blocks

`GA.py` is CPU-bound pure Python and already uses a `ProcessPoolExecutor`. When
launching several blocks at once, pin BLAS threads so the processes do not
oversubscribe cores:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

---

## 🧠 Model

`GA_LNN/lnn/model.py` — `LiquidCrossover`, **12,001 parameters** (52 KB checkpoint):

| Stage | Shape |
|---|---|
| `in_proj` | `Linear(3 → 48)` + ReLU |
| `liquid`, `liquid2` | two recurrent liquid layers, `d_model=48`, `liquid_steps=1` |
| `head` | `Linear(48 → 48)` → GELU → `Linear(48 → 1)` |

Each layer integrates `h ← h + (dt/τ)·(−h + tanh(W_x·x + W_h·h + b))` with a
learnable per-neuron time constant `τ = exp(log_tau) > 0`, applied token-wise
across customers with shared parameters.

- **Encoding** (`encoder.py`) — per customer: normalized rank in parent 1,
  normalized rank in parent 2, demand / capacity. Note that **no coordinate or
  distance information is supplied**, so instances with identical ranks and
  demands but different geometry are indistinguishable to the model.
- **Decoding** (`decoder.py`) — descending argsort of the scores, offset to
  1-based customer IDs.
- **Weights** — `GA_LNN/lnn/lnn_hyx.pt`. If missing or incompatible, `GA.py`
  prints a warning and proceeds with **random initialization** (`GA.py:90`).

Training (`GA_LNN/train_lnn.py`) is supervised imitation: the GA logs
`(parent1, parent2, child, instance)` triples produced by the classical
operators, and the network is trained to reproduce the child ordering.

---

## 📦 Output files

**Per-run metrics** — `results/<instance>_summary_<mut|nomut>.csv`

| Column | Meaning |
|---|---|
| `run` | row index within the file |
| `best_cost` | minimum Split cost in the final population |
| `avg_cost` | mean Split cost over the final population |
| `std_dev` | population SD of the final population — **not** SD across runs |
| `avg_exc` | `(avg_cost − best_known) / best_known × 100` |
| `time` | wall-clock seconds for that run |

**Convergence traces** — `results/convergence/<instance>/<op>/<mut|nomut>/run_<n>_convergence.csv`
with columns `generation, best_cost`.

**LNN training corpus** — `GA_LNN/data/lnn_log_<timestamp>_<uid>.pkl`, each a
pickled list of `(parent1, parent2, child, instance_name)` tuples with the depot
sentinels stripped. Flushed every 5,000 samples.

### Analysis scripts

| Script | Purpose |
|---|---|
| `ttest_vs_aex.py` | paired t-tests against the AEX baseline |
| `plot_k4_convergence.py` | convergence figures → `figures/` |
| `sensitivity_time_vs_best.py` | speed–quality scatter |
| `generate_logs.py`, `make_logs.sh` | corpus generation driver |
| `GA_LNN/lnn_arch_diagram.py` | architecture diagram |

---

## ⚠️ Status & known issues

Verified against the committed source. These are open defects, not design
choices, and they affect how the existing results should be read.

1. ~~**`python GA.py` crashes** with `KeyError: 'operator'` after completing the
   entire no-mutation sweep.~~ **Fixed** — `run_ga_for_instance` now returns the
   `"operator"` key that the aggregation step filters on (`GA.py:339`).

2. **`results/` does not match the current code.** The committed files are named
   `<instance>_<operator>_<regime>.csv`, but `write_metrics_to_csv` emits a
   single `<instance>_summary_<regime>.csv` per regime (`GA.py:376`). These
   results came from a different version of the pipeline and should be treated
   as audit-only.

3. **No selection pressure.** Parents are drawn with
   `random.randrange(len(fits))` (`GA.py:271`) — uniform random, ignoring
   fitness. The `fits` list is used only by `elitist_replacement`.

4. **Local search is not shared across encodings.** 2-opt runs only on the
   permutation path (`GA.py:290`). This makes the permutation/random-key
   comparison an unequal-compute comparison — in the committed results the
   random-key operators are ~1500× faster at `n=100` because they do far less
   work per generation, not because the operators are cheaper in kind.

5. **`std_dev` is not a run-level statistic.** It is `pstdev` over the 25
   individuals of the converged final population (`GA.py:334`), then averaged
   across runs. This is why several rows in `summary_table.csv` report exactly
   `0.0`.

6. **`BRKGA` is a misnomer.** `rk_brkga` (`GA.py:151`) is biased uniform
   random-key crossover inside the shared GA, not the BRKGA population
   algorithm (elite/non-elite selection, elite retention, mutants).

7. ~~**`Data/ACVRP/` is referenced but absent**, so `load_all_instances` raises
   `FileNotFoundError`.~~ **Fixed** — the 8 FTV1994 `.dat` files are now included,
   and missing instance files are skipped with a warning rather than raising
   (`cvrp_loader.py:154`). All 16 instances load.

   The depot-indexing defect this exposed (FTV1994 places the depot last, and
   the loader ignored `DEPOT_SECTION`, so all 8 instances loaded with customer 1
   acting as depot) is also fixed; `smoke_test.py` stage 2 guards it.

   Caveat: loading is not the same as validating. The best-known values in
   `cvrp_loader.py` are hardcoded and have not been checked against an
   independent route checker, and the asymmetric set has no `.sol` files. See
   the revision plan's validation step before using ACVRP results.

8. **2-opt optimizes the wrong objective — this is the most consequential
   defect found so far.** A GA chromosome carries the depot only at its ends,
   so `split_routes` (`GA.py:184`) returns a *single* route spanning all
   customers, and `two_opt_route` then improves it as an uncapacitated TSP
   tour. But `fitness` scores the chromosome with `split_cost`, which
   re-partitions it under capacity. The two objectives are different, and near
   good solutions they are anticorrelated.

   Demonstrated on the known optimum of `E-n22-k4`:

   | | TSP tour length | CVRP Split cost |
   |---|---|---|
   | reference solution (`.sol`, cost 375) | 339 | **375** |
   | after `two_opt_improvement` | 283 (better) | **436 (worse)** |

   On random tours the two objectives still correlate, so 2-opt helps early —
   across 30 random tours it never degraded Split cost. The damage appears only
   near the optimum, which is exactly where it matters: the GA cannot retain an
   optimal permutation, because its own local search would degrade it. This is
   consistent with every permutation operator plateauing at 377 and none ever
   reaching the reference 375.

   Independently verified as *not* a loader or evaluator problem: the reference
   routes cost exactly 375 under the loaded matrix, serve all 21 customers once
   within capacity, and `split_cost` on the reference giant tour returns 375.

   The fix is to apply 2-opt to the routes produced by Split, rather than to
   the giant tour. That changes search behaviour, so it is recorded here rather
   than applied.

---

## 📖 Citation

```bibtex
@software{guzey_lnn_crossover_cvrp,
  author  = {Alparslan Guzey},
  title   = {{AlparslanGuzey/LNN\_crossover\_CVRP}: Initial LNN crossover for CVRP},
  year    = {2025},
  month   = jun,
  doi     = {10.5281/zenodo.15762345},
  url     = {https://doi.org/10.5281/zenodo.15762345}
}
```

---

## 📜 License & acknowledgements

- Licensed under the MIT License.
- Built with PyTorch, NumPy, SciPy, pandas, and Matplotlib.
- Benchmark instances adapted from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/instances/1).
- The liquid layer is inspired by
  [Liquid Time-constant Networks (Hasani et al., 2020)](https://arxiv.org/abs/2006.04439);
  the implementation here is a custom Euler-style gated recurrence and does not
  reproduce that architecture exactly.

## 🧩 Contributions & issues

Pull requests are welcome. For bug reports or feature requests, please open an issue.
