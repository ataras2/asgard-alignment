#!/bin/bash

# Define the common parameters #"8298,2324"  "200,3000"
INITIAL_POS="2000,1000"
DX=50
DY=50
WIDTH=2000
HEIGHT=2000
ORIENTATION=0

# Loop through beams 1 to 4 and execute the Python script in parallel
for BEAM in {1..4}; do
    echo "Starting phasemask_raster.py for beam $BEAM..."
    python calibration/phasemask_raster.py --beam "$BEAM" \
        --initial_pos "$INITIAL_POS" --dx "$DX" --dy "$DY" \
        --width "$WIDTH" --height "$HEIGHT" --orientation "$ORIENTATION" &
done

# Wait for all background processes to complete
wait

echo "All beams processed in parallel."