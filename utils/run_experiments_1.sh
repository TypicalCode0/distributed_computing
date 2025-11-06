#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

PROG="task_1"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LAB_SRC="$DIR/../lab1/${PROG}.c"
BIN="$DIR/tmp_1/${PROG}"
results_1="$DIR/results_1/results_1_${PROG}_$(date +%Y%m%d_%H%M%S).csv"
PLOT_SCRIPT="$DIR/plot_results.py"

PROCS=(1 2 3 4 5 6 7 8)
SAMPLES=(1000000 1500000 10000000 15000000 20000000)
REPEATS=5

mkdir -p "$DIR/tmp_1"
mkdir -p "$DIR/results_1"
mkdir -p "$DIR/graphics_1"

if [ ! -f "$LAB_SRC" ]; then
  echo "Source $LAB_SRC not found" >&2
  exit 2
fi

mpicc "$LAB_SRC" -o "$BIN" || { echo "mpicc failed" >&2; exit 3; }

echo "proc,samples,run,time_sec,points_in_circle,pi_estimate" > "$results_1"

for proc in "${PROCS[@]}"; do
  for samples in "${SAMPLES[@]}"; do
    for ((r=1; r<=REPEATS; r++)); do

      echo "Running: proc=${proc} samples=${samples} run=${r}"

      if ! output=$(mpiexec -x PMIX_MCA_gds=hash -n "$proc" "$BIN" "$samples" 2>&1); then
        :
      fi

      time_sec=$(printf '%s\n' "$output" | grep -i "Total time" | sed -E 's/.*= *([0-9.]+).*/\1/' | head -n1 || true)

      if [[ -z "$time_sec" ]]; then
        tmp_1f=$(mktemp)
        /usr/bin/time -f "%e" -o "$tmp_1f" mpiexec -n "$proc" "$BIN" "$samples" > /dev/null 2>&1 || true
        time_sec=$(cat "$tmp_1f" 2>/dev/null || echo "")
        rm -f "$tmp_1f"
      fi
      [[ -z "$time_sec" ]] && time_sec="-1"

      points=$(printf '%s\n' "$output" | grep -i "Points in circle" | sed -E 's/.*: *([0-9]+).*/\1/' | head -n1 || true)
      [[ -z "$points" ]] && points="-1"

      pi_est=$(printf '%s\n' "$output" | grep -i "pi" | grep -Eo "[0-9]+\.[0-9]+" | head -n1 || true)
      [[ -z "$pi_est" ]] && pi_est="-1"

      echo "${proc},${samples},${r},${time_sec},${points},${pi_est}" >> "$results_1"

    done
  done
done

python3 "$PLOT_SCRIPT" --input "$results_1" --outdir "$DIR/graphics_1"
echo "find plots in $DIR/graphics_1"
