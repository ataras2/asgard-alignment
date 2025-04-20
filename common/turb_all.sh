#!/usr/bin/env bash
set -m   # enable job control

declare -A PIDS  # associative array: beam → PID

# Catch Ctrl‑C and clean up all children
trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# Launch the four beams
# --record_telem None <- this is default
for beam in 1 2 3 4; do
    python common/turbulence.py --V 3 --r0 0.3 --number_of_iterations 100000 \
    --number_of_modes_removed 0 --DM_chn 3 --max_time 360 --beam_id "$beam" &
    PIDS[$beam]=$!
    echo "Started beam $beam as PID ${PIDS[$beam]}"
done

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