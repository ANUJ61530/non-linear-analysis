#!/usr/bin/env bash
# Run nla_pipeline.py for every asset on the shared server, using at most max_workers_total worker processes
# (50% of the machine) at the lowest CPU priority. Finished stages are skipped, so the script can be rerun after an
# interruption. Usage:  bash run_all.sh            (all assets)
#                       bash run_all.sh BTC ETH    (selected assets)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../envs/nla/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
TOTAL=$($PY -c "import yaml; c=yaml.safe_load(open('config_server.yaml')); print(c['max_workers_total'])")
PER=$($PY -c "import yaml; c=yaml.safe_load(open('config_server.yaml')); print(c['jobs_per_asset'])")
PARALLEL=$(( TOTAL / PER ))
mkdir -p results/logs
if [ "$#" -gt 0 ]; then
  printf "%s\n" "$@" > results/logs/assets.txt
else
  $PY - > results/logs/assets.txt <<'PYEOF'
import pandas as pd, yaml
c = yaml.safe_load(open("config_server.yaml"))
panel = pd.read_parquet(f"{c['data_dir']}/nifty50/nifty50_clean_panel.parquet")
stocks = sorted(panel["adjclose"].columns)
print("\n".join(["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "TRX", "DOT", "NIFTY50"] + stocks))
PYEOF
fi
echo "$(date '+%F %T') start: $(wc -l < results/logs/assets.txt) assets, $PARALLEL at a time x $PER workers (max $TOTAL)" | tee -a results/logs/run_all.log
# the asset name is passed as an argument ($1), never pasted into the command, so names such as M&M are safe
xargs -a results/logs/assets.txt -d '\n' -P "$PARALLEL" -I{} \
  sh -c 'nice -n 19 "$0" nla_pipeline.py --asset "$1" --jobs "$2" > "results/logs/$1.log" 2>&1 && echo "$(date "+%F %T") ok $1" >> results/logs/run_all.log || echo "$(date "+%F %T") FAILED $1" >> results/logs/run_all.log' "$PY" {} "$PER"
echo "$(date '+%F %T') all done" | tee -a results/logs/run_all.log
