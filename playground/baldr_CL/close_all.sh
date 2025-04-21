#!/usr/bin/env bash
set -m   # enable job control


# ## incase in trouble:
# # PIDS=$(pgrep -f 'python .*baldr')
# # echo "Baldr PIDs: $PIDS"
# # # then when you’re ready:
# # kill $PIDS


# Default parameters (override via command‑line)
KP_LO=0.0; KI_LO=0.0;  KD_LO=0.0
KP_HO=0.0; KI_HO=0.0;  KD_HO=0.0
CAM_GAIN=1; CAM_FPS=200

usage() {
  cat <<EOF
Usage: $0 [--kp_LO <value>] [--ki_LO <value>] [--kd_LO <value>]
          [--kp_HO <value>] [--ki_HO <value>] [--kd_HO <value>]
          [--cam_gain <value>] [--cam_fps <value>]

Options:
  --kp_LO     Proportional gain for the low‑order controller (default: $KP_LO)
  --ki_LO     Integral gain for the low‑order controller (default: $KI_LO)
  --kd_LO     Derivative gain for the low‑order controller (default: $KD_LO)
  --kp_HO     Proportional gain for the high‑order controller (default: $KP_HO)
  --ki_HO     Integral gain for the high‑order controller (default: $KI_HO)
  --kd_HO     Derivative gain for the high‑order controller (default: $KD_HO)
  --cam_gain  Camera gain setting (default: $CAM_GAIN)
  --cam_fps   Camera frame rate in Hz (default: $CAM_FPS)
  -h, --help  Show this help message and exit
EOF
  exit 1
}

# Parse command‑line args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kp_LO)    KP_LO="$2";    shift 2 ;;
    --ki_LO)    KI_LO="$2";    shift 2 ;;
    --kd_LO)    KD_LO="$2";    shift 2 ;;
    --kp_HO)    KP_HO="$2";    shift 2 ;;
    --ki_HO)    KI_HO="$2";    shift 2 ;;
    --kd_HO)    KD_HO="$2";    shift 2 ;;
    --cam_gain) CAM_GAIN="$2"; shift 2 ;;
    --cam_fps)  CAM_FPS="$2";  shift 2 ;;
    -h|--help)  usage ;;
    *)          echo "Unknown option: $1" >&2; usage ;;
  esac
done

declare -A PIDS

# Cleanup function: kills all background jobs and their children
cleanup() {
  echo
  echo "Cleaning up all beams…"
  local j pid
  for pid in $(jobs -p); do
    # kill any children of this job
    pkill -TERM -P "$pid" 2>/dev/null
    # kill the job itself
    kill -TERM "$pid" 2>/dev/null
  done
  # wait for all jobs to exit
  wait
}

# Trap on exit or Ctrl‑C
trap 'cleanup; exit' SIGINT SIGTERM EXIT

# Launch four instances of CL.py, one per beam
for beam in 1 2 3 4; do
  python playground/baldr_CL/CL.py \
    --cam_gain "$CAM_GAIN" \
    --cam_fps  "$CAM_FPS"  \
    --phasemask "H3"       \
    --number_of_iterations 3001 \
    --kp_LO  "$KP_LO"      \
    --ki_LO  "$KI_LO"      \
    --kd_LO  "$KD_LO"      \
    --kp_HO  "$KP_HO"      \
    --ki_HO  "$KI_HO"      \
    --kd_HO  "$KD_HO"      \
    --beam_id "$beam" &
  PIDS[$beam]=$!
  echo "Started beam $beam → PID ${PIDS[$beam]}"
done

# Interactive commands
cat <<EOF

Commands:
  pause <n>    → SIGSTOP  beam n
  resume <n>   → SIGCONT  beam n
  stop <n>     → SIGTERM  beam n
  status       → list all beams & PIDs
  quit         → terminate everything & exit

EOF

while true; do
  read -p "> " cmd arg
  case "$cmd" in
    pause)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGSTOP "${PIDS[$arg]}" && echo "Paused beam $arg."
      else
        echo "No such beam: $arg"
      fi
      ;;
    resume)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGCONT "${PIDS[$arg]}" && echo "Resumed beam $arg."
      else
        echo "No such beam: $arg"
      fi
      ;;
    stop)
      if [[ -n ${PIDS[$arg]} ]]; then
        kill -SIGTERM "${PIDS[$arg]}" && echo "Stopping beam $arg."
        unset PIDS[$arg]
      else
        echo "No such beam: $arg"
      fi
      ;;
    status)
      for b in "${!PIDS[@]}"; do
        printf "Beam %s: PID %s (%s)\n" \
          "$b" "${PIDS[$b]}" \
          "$(ps -o state= -p "${PIDS[$b]}" 2>/dev/null||echo dead)"
      done
      ;;
    quit)
      cleanup
      exit
      ;;
    *)
      echo "Unknown command: $cmd"
      ;;
  esac
done

# #!/usr/bin/env bash
# set -m   # enable job control


# ## incase in trouble:
# # PIDS=$(pgrep -f 'python .*baldr')
# # echo "Baldr PIDs: $PIDS"
# # # then when you’re ready:
# # kill $PIDS

# # Default parameters (override via command‑line)
# KP_LO=0.0
# KI_LO=0.05
# KD_LO=0.0
# KP_HO=0.0
# KI_HO=0.02
# KD_HO=0.0
# CAM_GAIN=1
# CAM_FPS=200

# usage() {
#     cat <<EOF
# Usage: $0 [--kp_LO <value>] [--ki_LO <value>] [--kd_LO <value>]
#           [--kp_HO <value>] [--ki_HO <value>] [--kd_HO <value>]
#           [--cam_gain <value>] [--cam_fps <value>]

# Options:
#   --kp_LO     Proportional gain for the low‑order controller (default: $KP_LO)
#   --ki_LO     Integral gain for the low‑order controller (default: $KI_LO)
#   --kd_LO     Derivative gain for the low‑order controller (default: $KD_LO)
#   --kp_HO     Proportional gain for the high‑order controller (default: $KP_HO)
#   --ki_HO     Integral gain for the high‑order controller (default: $KI_HO)
#   --kd_HO     Derivative gain for the high‑order controller (default: $KD_HO)
#   --cam_gain  Camera gain setting (default: $CAM_GAIN)
#   --cam_fps   Camera frame rate in Hz (default: $CAM_FPS)
#   -h, --help  Show this help message and exit
# EOF
#     exit 1
# }

# # --- parse command‑line args ---
# while [[ $# -gt 0 ]]; do
#     case "$1" in
#         --kp_LO)     KP_LO="$2";    shift 2 ;;
#         --ki_LO)     KI_LO="$2";    shift 2 ;;
#         --kd_LO)     KD_LO="$2";    shift 2 ;;
#         --kp_HO)     KP_HO="$2";    shift 2 ;;
#         --ki_HO)     KI_HO="$2";    shift 2 ;;
#         --kd_HO)     KD_HO="$2";    shift 2 ;;
#         --cam_gain)  CAM_GAIN="$2"; shift 2 ;;
#         --cam_fps)   CAM_FPS="$2";  shift 2 ;;
#         -h|--help)   usage ;;
#         *)  echo "Unknown option: $1" >&2; usage ;;
#     esac
# done

# declare -A PIDS  # beam_id → PID

# # clean‑up on Ctrl‑C
# trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# # Launch four instances of CL.py, one per beam
# for beam in 1 2 3 4; do
#     python playground/baldr_CL/CL.py \
#         --cam_gain "$CAM_GAIN" \
#         --cam_fps "$CAM_FPS" \
#         --phasemask "H3" \
#         --number_of_iterations 3001 \
#         --kp_LO "$KP_LO" \
#         --ki_LO "$KI_LO" \
#         --kd_LO "$KD_LO" \
#         --kp_HO "$KP_HO" \
#         --ki_HO "$KI_HO" \
#         --kd_HO "$KD_HO" \
#         --beam_id "$beam" &
#     PIDS[$beam]=$!
#     echo "Started beam $beam as PID ${PIDS[$beam]} (kp_LO=$KP_LO, ki_LO=$KI_LO, kd_LO=$KD_LO, kp_HO=$KP_HO, ki_HO=$KI_HO, kd_HO=$KD_HO, cam_gain=$CAM_GAIN, cam_fps=$CAM_FPS)"
# done

# # #!/usr/bin/env bash
# # set -m   # enable job control

# # # Default gains (can be overridden via command‑line)
# # KI_LO=0.05
# # KI_HO=0.02

# # usage() {
# #     cat <<EOF
# # Usage: $0 [--ki_LO <value>] [--ki_HO <value>]

# # Options:
# #   --ki_LO    Integral gain for the low‑order controller (default: $KI_LO)
# #   --ki_HO    Integral gain for the high‑order controller (default: $KI_HO)
# #   -h, --help Show this help message and exit
# # EOF
# #     exit 1
# # }

# # # --- parse command‑line args ---
# # while [[ $# -gt 0 ]]; do
# #     case "$1" in
# #         --ki_LO)
# #             KI_LO="$2"; shift 2 ;;
# #         --ki_HO)
# #             KI_HO="$2"; shift 2 ;;
# #         -h|--help)
# #             usage ;;
# #         *)
# #             echo "Unknown option: $1" >&2
# #             usage ;;
# #     esac
# # done

# declare -A PIDS  # beam_id → PID

# # clean‑up on Ctrl‑C
# trap 'echo; echo "Shutting down all beams…"; kill "${PIDS[@]}" &>/dev/null; exit' SIGINT

# # Launch four instances of CL.py, one per beam
# for beam in 1 2 3 4; do
#     python playground/baldr_CL/CL.py \
#         --cam_gain 1 \
#         --cam_fps 200 \
#         --phasemask "H3" \
#         --number_of_iterations 3001 \
#         --ki_LO "$KI_LO" \
#         --ki_HO "$KI_HO" \
#         --beam_id "$beam" &
#     PIDS[$beam]=$!
#     echo "Started beam $beam as PID ${PIDS[$beam]} (ki_LO=$KI_LO, ki_HO=$KI_HO)"
# done

# echo
# echo "Commands:"
# echo "  pause <n>    → SIGSTOP  beam n"
# echo "  resume <n>   → SIGCONT  beam n"
# echo "  stop <n>     → SIGTERM  beam n"
# echo "  status       → list all beams & PIDs"
# echo "  quit         → terminate everything & exit"
# echo

# # Interactive loop
# while true; do
#   read -p "> " cmd arg
#   case $cmd in
#     pause)
#       if [[ -n ${PIDS[$arg]} ]]; then
#         kill -SIGSTOP ${PIDS[$arg]} && echo "Beam $arg paused."
#       else
#         echo "No such beam: $arg"
#       fi
#       ;;
#     resume)
#       if [[ -n ${PIDS[$arg]} ]]; then
#         kill -SIGCONT ${PIDS[$arg]} && echo "Beam $arg resumed."
#       else
#         echo "No such beam: $arg"
#       fi
#       ;;
#     stop)
#       if [[ -n ${PIDS[$arg]} ]]; then
#         kill -SIGTERM ${PIDS[$arg]} && echo "Beam $arg stopping."
#         unset PIDS[$arg]
#       else
#         echo "No such beam: $arg"
#       fi
#       ;;
#     status)
#       for b in "${!PIDS[@]}"; do
#         printf "Beam %s: PID %s  (%s)\n" \
#           "$b" "${PIDS[$b]}" \
#           "$(ps -o state= -p ${PIDS[$b]} 2>/dev/null || echo 'dead')"
#       done
#       ;;
#     quit)
#       echo "Terminating all beams…"
#       kill "${PIDS[@]}" &>/dev/null
#       exit
#       ;;
#     *)
#       echo "Unknown command: $cmd"
#       ;;
#   esac
# done