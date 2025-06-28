#!/usr/bin/env bash
set -e

INST_DIR="Data"                # folder that holds ACVRP / SCVRP trees
LOG_DIR="GA_LNN/data"
mkdir -p "$LOG_DIR"

echo "🔎 Discovering .vrp files under \$INST_DIR …"
mapfile -t VRP_FILES < <(find "$INST_DIR" -type f -name "*.vrp" | sort)

for vrp in "${VRP_FILES[@]}"; do
    inst_name=$(basename "${vrp%.*}")   # strip directory & .vrp
    echo -e "\n🚀 Running GA logger on instance: $inst_name"
    python GA.py "$inst_name"
done

echo -e "\n✅ Done – pickles now in \$LOG_DIR"
