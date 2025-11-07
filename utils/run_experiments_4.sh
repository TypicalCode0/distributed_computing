#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# run_experiments_4.sh - minimal console output
# Prints only: "Running: proc=... samples=... run=..."
# Uses only program's output to get time and iterations (no external /usr/bin/time).
# Saves raw program output into logs and results into CSV.

PROG="task_4"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LAB_SRC="$DIR/../lab1/${PROG}.c"
BIN="$DIR/tmp_4/${PROG}"
RESULTS_DIR="$DIR/results_4"
RESULTS="$RESULTS_DIR/results_${PROG}_$(date +%Y%m%d_%H%M%S).csv"
LOGDIR="$DIR/logs"
# Optional plot script is not invoked to keep console minimal
PLOT_SCRIPT="$DIR/plot_results.py"

# Experiment params (override via env)
PROCS=(1 4 8)
SIZES=(64 128 256)
REPEATS=3
GRID_C="${GRID_C:-0.0}"     # border_val
MAX_ITER="${MAX_ITER:-100000}"

mkdir -p "$DIR/tmp_4"
mkdir -p "$RESULTS_DIR"
mkdir -p "$DIR/graphics_4"
mkdir -p "$LOGDIR"

if [ ! -f "$LAB_SRC" ]; then
  echo "Source $LAB_SRC not found" >&2
  exit 2
fi

# compile
mpicc "$LAB_SRC" -lm -o "$BIN" || { echo "mpicc failed" >&2; exit 3; }

# detect MPI launcher (array form to avoid 'command not found' issues)
if command -v mpiexec >/dev/null 2>&1; then
  MPI_LAUNCH_COMMAND=("mpiexec" "-x" "PMIX_MCA_gds=hash" "-n")
elif command -v mpirun >/dev/null 2>&1; then
  MPI_LAUNCH_COMMAND=("mpirun" "-n")
else
  echo "No mpiexec or mpirun found in PATH. Please install OpenMPI/MPICH or set MPI launcher." >&2
  exit 4
fi

# robust parser: returns "time|iter" (handles russian/english variants)
parse_time_iter() {
  local out="$1"
  local time_s=""
  local iter=""

  time_s="$(printf "%s\n" "$out" | grep -Eo 'time[[:space:]]*=[[:space:]]*[0-9]+(\.[0-9]+)?' | head -n1 | sed -E 's/.*=//; s/ //g' || true)"
  if [ -z "$time_s" ]; then
    time_s="$(printf "%s\n" "$out" | grep -iEo 'Time:[[:space:]]*[0-9]+(\.[0-9]+)?' | head -n1 | sed -E 's/.*: *([0-9.]+).*/\1/' || true)"
  fi
  if [ -z "$time_s" ]; then
    time_s="$(printf "%s\n" "$out" | grep -iEo '[0-9]+(\.[0-9]+)?[[:space:]]*(секунд|seconds)' | head -n1 | grep -Eo '[0-9]+(\.[0-9]+)?' || true)"
  fi
  if [ -z "$time_s" ]; then
    time_s="$(printf "%s\n" "$out" | grep -Eio 'time[[:space:]=]*[0-9]+(\.[0-9]+)?' | head -n1 | sed -E 's/.*[^0-9]*([0-9.]+).*/\1/' || true)"
  fi

  iter="$(printf "%s\n" "$out" | grep -Eo 'Сошлась после [0-9]+' | grep -Eo '[0-9]+' || true)"
  if [ -z "$iter" ]; then
    iter="$(printf "%s\n" "$out" | grep -Eo 'Convergence reached after [0-9]+' | grep -Eo '[0-9]+' || true)"
  fi
  if [ -z "$iter" ]; then
    iter="$(printf "%s\n" "$out" | grep -Eo 'Converged in [0-9]+' | grep -Eo '[0-9]+' || true)"
  fi

  [ -z "$time_s" ] && time_s="-1"
  [ -z "$iter" ] && iter="-1"

  printf "%s|%s" "$time_s" "$iter"
}

# CSV header
echo "samples,proc,time_sec,run,iter" > "$RESULTS"

# Main experiment loop
for size in "${SIZES[@]}"; do
  for proc in "${PROCS[@]}"; do
    for ((r=1; r<=REPEATS; r++)); do
      # Minimal console output exactly as requested
      echo "Running: proc=${proc} samples=${size} run=${r}"

      LOGFILE="${LOGDIR}/${PROG}_size${size}_p${proc}_run${r}.log"

      # Run MPI job; capture output but do NOT print it to console
      out="$("${MPI_LAUNCH_COMMAND[@]}" "$proc" "$BIN" "$size" "$GRID_C" "$MAX_ITER" 2>&1 || true)"

      # Save raw output to log (for debugging later)
      printf "%s\n" "$out" > "$LOGFILE"

      # Parse time and iter strictly from program output
      parsed="$( parse_time_iter "$out" )"
      time_s="$(printf "%s" "$parsed" | awk -F'|' '{print $1}')"
      iter="$(printf "%s" "$parsed" | awk -F'|' '{print $2}')"

      # Write results to CSV
      echo "${size},${proc},${time_s},${r},${iter}" >> "$RESULTS"

      # tiny pause
      sleep 0.1
    done
  done
done

# Final short message
echo "Done. Results saved to: $RESULTS"
