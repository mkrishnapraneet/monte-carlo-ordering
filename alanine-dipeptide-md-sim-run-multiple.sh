#!/usr/bin/env bash
# run_ensemble.sh
# Launch N independent alanine dipeptide MD runs in parallel.
#
# Usage:
#   bash run_ensemble.sh [N_RUNS] [PROD_NS] [EQUIL_NS] [OUTDIR]
#
# Defaults:
#   N_RUNS  = 4
#   PROD_NS = 2.0
#   EQUIL_NS= 0.2
#   OUTDIR  = alanine-dipeptide
#
# Each run gets a unique, randomly drawn --seed and --solv_seed so that
# velocity initialisation, solvation geometry, and padding all differ.
#
# Logs for each run are tee-d to <OUTDIR>/run_<i>/launcher.log
# so you can follow progress with:
#   tail -f alanine-dipeptide/launcher_0.log

set -euo pipefail

N_RUNS="${1:-4}"
PROD_NS="${2:-2.0}"
EQUIL_NS="${3:-0.2}"
OUTDIR="${4:-alanine-dipeptide}"

SCRIPT="$(dirname "$0")/ala_run.py"

if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: ala_run.py not found at $SCRIPT"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "============================================"
echo "  Alanine dipeptide ensemble launcher"
echo "  Runs    : $N_RUNS"
echo "  Prod    : ${PROD_NS} ns / run"
echo "  Equil   : ${EQUIL_NS} ns / run"
echo "  Output  : $OUTDIR"
echo "============================================"

PIDS=()

for (( i=0; i<N_RUNS; i++ )); do
    # draw independent random seeds (32-bit positive integers)
    SEED=$(python3 -c "import random; print(random.randint(0, 2**31-1))")
    SOLV_SEED=$(python3 -c "import random; print(random.randint(0, 2**31-1))")

    RUN_DIR="${OUTDIR}"
    mkdir -p "$RUN_DIR"

    LOG="${OUTDIR}/launcher_${i}.log"

    echo "Starting run $i  (seed=$SEED  solv_seed=$SOLV_SEED)  → $LOG"

    python3 "$SCRIPT" \
        --index      "$i"        \
        --outdir     "$OUTDIR"   \
        --seed       "$SEED"     \
        --solv_seed  "$SOLV_SEED"\
        --prod_ns    "$PROD_NS"  \
        --equil_ns   "$EQUIL_NS" \
        > >(tee "$LOG") 2>&1 &

    PIDS+=($!)
done

echo ""
echo "All $N_RUNS runs launched (PIDs: ${PIDS[*]})"
echo "Waiting for all to finish ..."
echo ""

# track each PID and report exit status
FAILED=0
for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if wait "$pid"; then
        echo "  run $i (PID $pid): DONE ✓"
    else
        echo "  run $i (PID $pid): FAILED ✗"
        (( FAILED++ )) || true
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All $N_RUNS runs completed successfully."
else
    echo "$FAILED / $N_RUNS runs failed. Check the launcher.log files."
    exit 1
fi

echo ""
echo "Output layout (all files flat in ${OUTDIR}/):"
echo "  input.pdb                     ← shared input (written once)"
echo "  solvated_<i>.pdb"
echo "  prod_<i>.dcd                  ← raw (wrapped) trajectory"
echo "  prod_<i>.log"
echo "  prod_<i>.chk"
echo "  final_<i>.pdb"
echo "  unwrapped_<i>.dcd             ← solute-only, unwrapped"
echo "  unwrapped_<i>.pdb             ← last frame, solute-only"
echo "  launcher_<i>.log"