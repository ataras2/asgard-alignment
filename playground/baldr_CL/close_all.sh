# #!/bin/bash
# # Run the CL.py script for beam_id 1, 2, 3, and 4 concurrently

# # for beam in 1 2 3 4; do
# #     #python /home/asg/Progs/repos/asgard-alignment/playground/baldr_CL/CL.py --number_of_iterations 1000 --beam_id "$beam" &
# #     python playground/baldr_CL/CL.py  --cam_gain 10 --cam_fps 1000 --phasemask "H3" --number_of_iterations 3001 --beam_id "&beam" &
# # done

# for beam in 1 2 3 4; do
#     python playground/baldr_CL/CL.py \
#         --cam_gain 10 \
#         --cam_fps 1000 \
#         --phasemask "H3" \
#         --number_of_iterations 3001 \
#         --beam_id "$beam" &        # note: "$beam", not "&beam"
# done

# # Wait for all background processes to complete
# wait

# echo "All beam processes have finished."



#!/usr/bin/env bash
set -m   # enable job control

declare -A PIDS  # associative array: beam → PID

# Catch Ctrl‑C and clean up all children
trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# Launch the four beams
for beam in 1 2 3 4; do #3
    python playground/baldr_CL/CL.py \
      --cam_gain 10 --cam_fps 1000 --phasemask "H3" \
      --number_of_iterations 3001 --beam_id "$beam" &
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