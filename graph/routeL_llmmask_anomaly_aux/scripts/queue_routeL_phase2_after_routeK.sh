#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ROUTEK_PID="${1:?Route K PID required}"
OUTDIR="${2:?Output dir required}"
LOG_PREFIX="${3:?Log prefix required}"

mkdir -p graph/logs
RUNLOG="graph/logs/${LOG_PREFIX}.log"
echo "[$(date '+%F %T')] Route L queue started" >> "$RUNLOG"
echo "[$(date '+%F %T')] waiting for Route K pid ${ROUTEK_PID}" >> "$RUNLOG"
while ps -p "${ROUTEK_PID}" >/dev/null 2>&1; do
  sleep 60
done

echo "[$(date '+%F %T')] Route K finished, waiting for GPU compute idle" >> "$RUNLOG"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '^[0-9]'; do
  sleep 20
done

echo "[$(date '+%F %T')] GPU idle, starting Route L Phase 2 -> ${OUTDIR}" >> "$RUNLOG"
python3 -m graph.routeL_llmmask_anomaly_aux.scripts.run_routeL_phase2 \
  --output_root "${OUTDIR}" \
  --seed 42 >> "$RUNLOG" 2>&1

if [[ -f "${OUTDIR}/FAILED_PRECHECK.md" ]]; then
  echo "[$(date '+%F %T')] precheck failed; skipping push" >> "$RUNLOG"
  exit 0
fi

if [[ ! -f "${OUTDIR}/routeL_summary.csv" ]]; then
  echo "[$(date '+%F %T')] routeL_summary.csv missing; skipping push" >> "$RUNLOG"
  exit 1
fi

echo "[$(date '+%F %T')] outputs ready, preparing git commit" >> "$RUNLOG"
git add graph/routeL_llmmask_anomaly_aux "${OUTDIR}"
if ! git diff --cached --quiet; then
  git commit -m "Add Route L Phase 2 anomaly fusion results" >> "$RUNLOG" 2>&1 || true
fi
git push origin main >> "$RUNLOG" 2>&1 || true
echo "[$(date '+%F %T')] push finished" >> "$RUNLOG"
