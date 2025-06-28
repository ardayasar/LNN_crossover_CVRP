# LNN Crossover for Capacitated Vehicle Routing Problem (CVRP)

**Repository**: `AlparslanGuzey/LNN_crossover_CVRP`  
**DOI**: [10.5281/zenodo.15762345](https://doi.org/10.5281/zenodo.15762345)

This repository contains an integrated framework for benchmarking classical and learned crossover operators—including a Liquid Neural Network (LNN)–based crossover—on the Capacitated Vehicle Routing Problem (CVRP). It supports both permutation‐based and random‐key encodings, with or without mutation, and produces detailed CSV logs for downstream analysis and supervised LNN training data.

---

## 🚀 Features

- **Eight Crossover Operators**  
  Permutation-based: `PMX`, `OX`, `CX`, `AEX`, `LNN`  
  Random-key–based: `BRKGA`, `RK-U`, `RK-GR`, `RK-2PX`

- **Mutation Regimes**  
  Benchmark runs with or without mutation. Supports:
  - `swap-2` mutation (for permutation encoding)
  - `bit-flip` mutation (for random-key encoding)

- **Automated CSV Logging**  
  - Per-run metrics in:  
    `results/<instance>_<operator>_<mut|nomut>.csv`  
  - Aggregated summary table:  
    `results/summary_table.csv`  
  - Optional LNN training logs in:  
    `GA_LNN/data/` (enabled via `LNN_LOG=1`)

- **LNN Crossover**  
  - Trained Liquid Neural Network model predicts crossover outputs  
  - Modular PyTorch implementation in `GA_LNN/lnn/`  
  - Supports learning new crossover strategies from data

- **Configurable Runtime**  
  Easily change:
  - `FEATURE_DIM`, `POP_SIZE`, `MAX_GENERATIONS`
  - `CROSSOVER_RATE`, `MUTATION_RATE`, `NUM_RUNS`
  - Target instances and selected operators

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlparslanGuzey/LNN_crossover_CVRP.git
   cd LNN_crossover_CVRP

	2.	Install dependencies

pip install -r requirements.txt


	3.	Prepare CVRP data
Place .vrp, .dat, and (optionally) .sol files under:

Data/SCVRP/
Data/ACVRP/

Update path logic in cvrp_loader.py if needed.

	4.	Download pretrained LNN weights
Ensure the model file exists at:

GA_LNN/lnn/lnn_hyx.pt

Alternatively, train your own model and place it here.

⸻

🧪 Usage

Run full benchmark:

python GA.py

Enable LNN training data logging:

LNN_LOG=1 python GA.py

Customize parameters in GA.py:

FEATURE_DIM     = 3
POP_SIZE        = 50
MAX_GENERATIONS = 1000
CROSSOVER_RATE  = 1.0
MUTATION_RATE   = 0.10
NUM_RUNS        = 30

Filter instances/operators:

TARGET_INSTANCES = {"E-n51-k5"}
ALLOWED_OPS      = {"OX", "CX", "AEX", "LNN", "BRKGA", "RK-2PX"}


⸻

📦 Output Files
	•	Per-instance Logs
results/<instance>_<op>_nomut.csv
results/<instance>_<op>_mut.csv
Columns:
run, best_cost, avg_cost, std_dev, avg_exc, time
	•	Summary Table
results/summary_table.csv
Columns:
instance, operator, mut, best, avg, std, avg_exc, time
	•	LNN Supervised Data (if enabled)
Stored as pickled tuples:
(parent1, parent2, child, instance)
Path: GA_LNN/data/

⸻

📖 Citation

If you use this codebase or the LNN crossover in your research, please cite:

Alparslan Guzey, “AlparslanGuzey/LNN_crossover_CVRP: Initial LNN crossover for CVRP”, Zenodo, Jun. 28, 2025.
DOI: 10.5281/zenodo.15762345

⸻

📜 License & Acknowledgements
	•	Licensed under the MIT License.
	•	Developed using PyTorch, NumPy, and standard Python tooling.
	•	Benchmark instances adapted from CVRPLIB.
	•	Inspired by LiquidCrossover and related neural genetic algorithm literature.

⸻

🧩 Contributions & Issues

Pull requests are welcome.
For bug reports or feature requests, please open an issue.
