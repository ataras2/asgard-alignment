#!/usr/bin/env bash
set -m   # enable job control

# Default parameters
V=0.6
R0=0.4
DM_CHN=3
MODES_REMOVED=0

usage() {
    cat <<EOF
Usage: $0 [--V <wind_speed>] [--r0 <Fried_param>] [--DM_chn <channel>] [--number_of_modes_removed <n>]

Options:
  --V                         Wind speed V (default: $V)
  --r0                        Fried parameter r0 (default: $R0)
  --DM_chn                    Number of DM channels (default: $DM_CHN)
  --number_of_modes_removed   Number of modes to remove (default: $MODES_REMOVED)
  -h, --help                  Show this help message and exit
EOF
    exit 1
}

# Parse command‑line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --V)
            V="$2"; shift 2 ;;
        --r0)
            R0="$2"; shift 2 ;;
        --DM_chn)
            DM_CHN="$2"; shift 2 ;;
        --number_of_modes_removed)
            MODES_REMOVED="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1" >&2
            usage ;;
    esac
done

declare -A PIDS  # beam → PID

# Catch Ctrl‑C and clean up all children
trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# Launch the four beams
for beam in 1 2 3 4; do
    python common/turbulence.py \
        --V "$V" \
        --r0 "$R0" \
        --number_of_iterations 100000 \
        --number_of_modes_removed "$MODES_REMOVED" \
        --DM_chn "$DM_CHN" \
        --max_time 360 \
        --beam_id "$beam" &
    PIDS[$beam]=$!
    echo "Started beam $beam as PID ${PIDS[$beam]} (V=$V, r0=$R0, DM_chn=$DM_CHN, modes_removed=$MODES_REMOVED)"
done

# #!/usr/bin/env bash
# set -m   # enable job control

# declare -A PIDS  # associative array: beam → PID

# # Catch Ctrl‑C and clean up all children
# trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# # Launch the four beams
# # --record_telem None <- this is default
# for beam in 1 2 3 4; do
#     python common/turbulence.py --V 0.6 --r0 0.4 --number_of_iterations 100000 \
#     --number_of_modes_removed 0 --DM_chn 3 --max_time 360 --beam_id "$beam" &
#     PIDS[$beam]=$!
#     echo "Started beam $beam as PID ${PIDS[$beam]}"
# done

echo
echo "Commands:"
echo "  pause <n>    → SIGSTOP  beam n"
echo "  resume <n>   → SIGCONT  beam n"
echo "  stop <n>     → SIGTERM  beam n"
echo "  status       → list all beams & PIDs"
echo "  quit         → terminate everything & exit"
echo

# Interactive loop
while true; do
  read -p "> " cmd arg
  case $cmd in
    pause)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGSTOP ${PIDS[$arg]} && echo "Beam $arg paused."
      else
        echo "No such beam: $arg"
      fi
      ;;
    resume)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGCONT ${PIDS[$arg]} && echo "Beam $arg resumed."
      else
        echo "No such beam: $arg"
      fi
      ;;
    stop)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGTERM ${PIDS[$arg]} && echo "Beam $arg stopping."
        unset PIDS[$arg]
      else
        echo "No such beam: $arg"
      fi
      ;;
    status)
      for b in "${!PIDS[@]}"; do
        printf "Beam %s: PID %s  (%s)\n" \
          "$b" "${PIDS[$b]}" \
          "$(ps -o state= -p ${PIDS[$b]} 2>/dev/null || echo 'dead')"
      done
      ;;
    quit)
      echo "Terminating all beams…"
      kill "${PIDS[@]}" &>/dev/null
      exit
      ;;
    *)
      echo "Unknown command: $cmd"
      ;;
  esac
done