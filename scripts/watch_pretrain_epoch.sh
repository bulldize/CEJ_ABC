#!/usr/bin/env sh

# Usage:
#   sh scripts/watch_pretrain_epoch.sh [metrics_csv_path] [sleep_seconds]
#
# Example:
#   sh scripts/watch_pretrain_epoch.sh /root/cej_runs/run_unsup_001/unsup/pretrain/metrics.csv 30

METRICS_PATH="${1:-/root/cej_runs/run_unsup_001/unsup/pretrain/metrics.csv}"
SLEEP_SEC="${2:-30}"

LAST_EPOCH=""

echo "watching metrics: ${METRICS_PATH}"
echo "poll interval: ${SLEEP_SEC}s"

# If a directory is passed, auto-append metrics.csv.
if [ -d "${METRICS_PATH}" ]; then
  METRICS_PATH="${METRICS_PATH%/}/metrics.csv"
  echo "resolved metrics path from dir: ${METRICS_PATH}"
fi

# If the given path still does not exist, try auto-discovery.
if [ ! -e "${METRICS_PATH}" ]; then
  AUTO_METRICS="$(find /root/cej_runs -type f -path '*/unsup/pretrain/metrics.csv' 2>/dev/null | head -n 1)"
  if [ -n "${AUTO_METRICS}" ]; then
    METRICS_PATH="${AUTO_METRICS}"
    echo "auto-detected metrics path: ${METRICS_PATH}"
  fi
fi

while true; do
  TS="$(date '+%F %T')"

  PROC_LINE="$(ps -eo pid,cmd | grep 'src.pretrain_mae' | grep -v grep | head -n 1)"
  if [ -n "${PROC_LINE}" ]; then
    TRAIN_STATE="RUNNING"
  else
    TRAIN_STATE="STOPPED"
  fi

  GPU_LINE="$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  if [ -n "${GPU_LINE}" ]; then
    GPU_UTIL="$(printf '%s' "${GPU_LINE}" | cut -d',' -f1 | xargs)"
    GPU_USED="$(printf '%s' "${GPU_LINE}" | cut -d',' -f2 | xargs)"
    GPU_TOTAL="$(printf '%s' "${GPU_LINE}" | cut -d',' -f3 | xargs)"
    GPU_INFO="gpu=${GPU_UTIL}% mem=${GPU_USED}/${GPU_TOTAL}MiB"
  else
    GPU_INFO="gpu=NA"
  fi

  if [ -f "${METRICS_PATH}" ]; then
    LINE="$(awk -F',' '
      { gsub(/\r/, "", $0) }
      NR > 1 && $1 ~ /^[0-9]+$/ && $2 ~ /^[-+0-9.eE]+$/ {last = $1 "," $2}
      END {print last}
    ' "${METRICS_PATH}")"

    if [ -n "${LINE}" ]; then
      EPOCH="${LINE%%,*}"
      LOSS="${LINE#*,}"

      if [ "${EPOCH}" != "${LAST_EPOCH}" ]; then
        STATS="$(awk -F',' -v cur_epoch="${EPOCH}" '
          { gsub(/\r/, "", $0) }
          NR > 1 && $1 ~ /^[0-9]+$/ && $2 ~ /^[-+0-9.eE]+$/ {
            n += 1
            e[n] = $1 + 0
            l[n] = $2 + 0
          }
          END {
            idx = -1
            for (i = 1; i <= n; i++) {
              if (e[i] == cur_epoch) {
                idx = i
              }
            }

            prev = "NA"
            avg5 = "NA"
            cnt5 = 0

            if (idx > 1) {
              prev = l[idx - 1]
              start = idx - 5
              if (start < 1) {
                start = 1
              }
              sum = 0
              cnt = 0
              for (j = start; j < idx; j++) {
                sum += l[j]
                cnt += 1
              }
              if (cnt > 0) {
                avg5 = sum / cnt
                cnt5 = cnt
              }
            }

            printf "%s,%s,%d\n", prev, avg5, cnt5
          }
        ' "${METRICS_PATH}")"

        PREV_LOSS="$(printf '%s' "${STATS}" | cut -d',' -f1)"
        AVG5_LOSS="$(printf '%s' "${STATS}" | cut -d',' -f2)"
        CNT5="$(printf '%s' "${STATS}" | cut -d',' -f3)"

        if [ "${PREV_LOSS}" = "NA" ]; then
          PREV_COMP="d_prev=NA"
        else
          PREV_COMP="$(awk -v cur="${LOSS}" -v prev="${PREV_LOSS}" '
            BEGIN {
              d = cur - prev
              p = (prev != 0 ? d / prev * 100 : 0)
              printf "d_prev=%+.6f (%+.3f%%)", d, p
            }
          ')"
        fi

        if [ "${AVG5_LOSS}" = "NA" ]; then
          AVG5_COMP="d_avg5=NA"
        else
          AVG5_COMP="$(awk -v cur="${LOSS}" -v avg5="${AVG5_LOSS}" -v n="${CNT5}" '
            BEGIN {
              d = cur - avg5
              p = (avg5 != 0 ? d / avg5 * 100 : 0)
              printf "d_avg5(%d)=%+.6f (%+.3f%%)", n, d, p
            }
          ')"
        fi

        echo "${TS} | epoch=${EPOCH} loss=${LOSS} | ${PREV_COMP} | ${AVG5_COMP} | ${GPU_INFO} | train=${TRAIN_STATE} | NEW_EPOCH"
        LAST_EPOCH="${EPOCH}"
      else
        echo "${TS} | epoch=${EPOCH} loss=${LOSS} | no_new_epoch | ${GPU_INFO} | train=${TRAIN_STATE}"
      fi
    else
      echo "${TS} | epoch=NA | no_metrics_line | ${GPU_INFO} | train=${TRAIN_STATE}"
    fi
  else
    echo "${TS} | epoch=NA | metrics_missing | ${GPU_INFO} | train=${TRAIN_STATE}"
  fi

  sleep "${SLEEP_SEC}"
done
