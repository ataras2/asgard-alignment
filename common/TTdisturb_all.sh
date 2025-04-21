#!/usr/bin/env bash
set -m   # enable job control

# Default values
DT=0.05
DM_RMS=0.02
DIST_TYPE="vib"
VIB_FREQ=1

usage() {
    cat <<EOF
Usage: $0 [--dt <seconds>] [--dm_rms <rms_value>] [--dist_type <vib|1onf>] [--vib_freq <Hz>]

Options:
  --dt         Time step between iterations (default: $DT)
  --dm_rms     RMS amplitude for DM disturbance (default: $DM_RMS)
  --dist_type  Disturbance type: "vib" or "1onf" (default: $DIST_TYPE)
  --vib_freq   Vibration frequency in Hz (default: $VIB_FREQ)
  -h, --help   Show this help message and exit
EOF
    exit 1
}

# Parse command‑line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dt)        DT="$2"; shift 2 ;;
        --dm_rms)    DM_RMS="$2"; shift 2 ;;
        --dist_type) DIST_TYPE="$2"; shift 2 ;;
        --vib_freq)  VIB_FREQ="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *)           echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Single declaration of PIDS
declare -A PIDS  # beam_id → PID

# Trap both Ctrl‑C and a normal 'kill'
cleanup(){
  echo
  echo "Shutting down all beams…"
  kill "${PIDS[@]}" &>/dev/null
  # as a fallback, kill any stray TT_disturb.py
  pkill -f TT_disturb.py &>/dev/null || true
  exit
}
trap cleanup SIGINT SIGTERM

# Launch the four beams once
for beam in 1 2 3 4; do
    python common/TT_disturb.py \
        --dt "$DT" \
        --dm_rms "$DM_RMS" \
        --dist_type "$DIST_TYPE" \
        --vib_freq "$VIB_FREQ" \
        --number_of_iterations 100000 \
        --DM_chn 3 \
        --max_time 360 \
        --beam_id "$beam" &
    PIDS[$beam]=$!
    echo "Started beam $beam as PID ${PIDS[$beam]}"
done

echo
echo "Interactive commands:"
echo "  pause <n>    → SIGSTOP  beam n"
echo "  resume <n>   → SIGCONT  beam n"
echo "  stop <n>     → SIGTERM  beam n"
echo "  status       → list all beams & PIDs"
echo "  quit         → terminate everything & exit"
echo

# Interactive control loop
while true; do
  read -r -p "> " cmd arg
  case $cmd in
    pause)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -STOP ${PIDS[$arg]} && echo "Beam $arg paused."
      else
        echo "No such beam: $arg"
      fi
      ;;
    resume)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -CONT ${PIDS[$arg]} && echo "Beam $arg resumed."
      else
        echo "No such beam: $arg"
      fi
      ;;
    stop)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill ${PIDS[$arg]} && echo "Beam $arg stopping."
        unset PIDS[$arg]
      else
        echo "No such beam: $arg"
      fi
      ;;
    status)
      for b in "${!PIDS[@]}"; do
        state=$(ps -o state= -p "${PIDS[$b]}" 2>/dev/null || echo "dead")
        printf "Beam %s: PID %s  (%s)\n" "$b" "${PIDS[$b]}" "$state"
      done
      ;;
    quit)
      cleanup
      ;;
    *)
      echo "Unknown command: $cmd"
      ;;
  esac
done