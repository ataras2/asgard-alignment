#!/bin/bash
# Run the CL.py script for beam_id 1, 2, 3, and 4 concurrently

for beam in 1 2 3 4; do
    python /home/asg/Progs/repos/asgard-alignment/playground/baldr_CL/CL.py --number_of_iterations 1000 --beam_id "$beam" &
done

# Wait for all background processes to complete
wait

echo "All beam processes have finished."