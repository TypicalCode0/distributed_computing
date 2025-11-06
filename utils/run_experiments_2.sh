#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

PROG="task_2"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LAB_SRC="$DIR/../lab1/${PROG}.c"
BIN="$DIR/tmp_2/${PROG}"
RESULTS="$DIR/results_2/results_${PROG}_$(date +%Y%m%d_%H%M%S).csv"
PLOT_SCRIPT="$DIR/plot_results.py"

PROCS=(1 2 3 4 5 6 7 8)
SIZES=(1000 10000 20000)
MODES=(row col block)
REPEATS=5

mkdir -p "$DIR/tmp_2" "$DIR/results_2" "$DIR/graphics_2"

[ -f "$LAB_SRC" ] || { echo "Source $LAB_SRC not found" >&2; exit 1; }

mpicc "$LAB_SRC" -o "$BIN" || { echo "mpicc failed" >&2; exit 2; }

echo "mode,proc,rows,cols,run,time_sec" > "$RESULTS"

for mode in "${MODES[@]}"; do
  for size in "${SIZES[@]}"; do
    for proc in "${PROCS[@]}"; do
      for ((r=1; r<=REPEATS; r++)); do
        echo "Running: mode=${mode} size=${size} proc=${proc} run=${r}"
        output=$(mpiexec -x PMIX_MCA_gds=hash -n "$proc" "$BIN" "$mode" "$size" 2>&1 || true)

        time_sec=$(printf '%s\n' "$output" | grep -i "Total time" | sed -E 's/.*= *([0-9.]+).*/\1/' | head -n1 || true)

        if [[ -z "$time_sec" ]]; then
          tmpf=$(mktemp)
          /usr/bin/time -f "%e" -o "$tmpf" mpiexec -n "$proc" "$BIN" "$mode" "$size" >/dev/null 2>&1 || true
          time_sec=$(cat "$tmpf" 2>/dev/null || echo "")
          rm -f "$tmpf"
        fi
        [[ -z "$time_sec" ]] && time_sec="-1"

        echo "${mode},${proc},${size},${size},${r},${time_sec}" >> "$RESULTS"
      done
    done
  done
done

python3 "$PLOT_SCRIPT" --input "$RESULTS" --outdir "$DIR/graphics_2" || echo "plotting failed"
echo "plots: $DIR/graphics_2  results: $RESULTS"
