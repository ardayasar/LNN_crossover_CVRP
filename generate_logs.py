# generate_logs.py
import os, glob, subprocess, sys, random

# Target number of log files to generate
TARGET_PKL = 20_000          # ⟵ stop once this many log files exist

# 1) where your .vrp / .dat live:
ROOT     = os.path.dirname(__file__)
INST_DIR = os.path.join(ROOT, "Data")
LOG_DIR  = os.path.join(ROOT, "GA_LNN", "data")
os.makedirs(LOG_DIR, exist_ok=True)

# Count how many *.pkl files we already have
def pkl_count() -> int:
    return len(glob.glob(os.path.join(LOG_DIR, "lnn_log_*.pkl")))

# 2) find all instances:
paths = (
    glob.glob(os.path.join(INST_DIR, "ACVRP", "*.dat")) +
    glob.glob(os.path.join(INST_DIR, "SCVRP", "*.vrp"))
)
instances = sorted({ os.path.splitext(os.path.basename(p))[0] for p in paths })

# shuffle order for diversity
random.shuffle(instances)

total = pkl_count()
print(f"🔎 Found {len(instances)} candidate instances, existing logs: {total}")

# 3) loop until we hit TARGET_PKL
while total < TARGET_PKL:
    for inst in instances:
        if total >= TARGET_PKL:
            break
        print(f"🚀 logging instance {inst}  (current .pkl files: {total}/{TARGET_PKL})")
        # call GA.py with LNN logging turned on
        env = dict(os.environ, LNN_LOG="1")
        subprocess.run(
            [sys.executable, os.path.join(ROOT, "GA.py"), inst],
            check=True,
            env=env
        )
        total = pkl_count()

print(f"\n✅ Reached {total} pickle files – stopping.  Logs saved in {LOG_DIR}")