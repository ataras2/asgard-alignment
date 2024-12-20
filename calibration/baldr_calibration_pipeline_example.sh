


# register the Baldr pupil sub regions to crop 
python calibration/detect_cropped_pupils_coords.py

# acquire the phasemasks for a given beam 
python calibration/phasemask_routine.py --search_radius 200 --step_size 20 --phasemask_name H3 --beam 2

# take clear (phasemask out) and phasemask in pupils (editing) over series of FPS
# python playground/baldr_series_reference_pupil_intensities.py --beam 2 --phasemask H3 --no_frames 1000 --cam_gain 1 

# pokeramp : poke each DM actuator over a +/- range and record the images 
python calibration/pokeramps.py --phasemask_name H3 --cam_gain 5 --cam_fps 50

# apply a rolling Kolmogorov phase screen on all the DMs and record images 
python calibration/kolmogorov_phasescreen_on_dm.py --phasemask_name H3 --cam_gain 5 --cam_fps 50

# use the most recent pokeramp file and (optionally) data from the 
# rolling Kolmogorov phase screen on the DM to calibrate control models
# for Baldr  register DM in the detector plane.
# Also write an PDF report on the calibration
python calibration/baldr_calibration.py recent --kol recent --beam 2 --write_report True





#### STABILITY 
# #takes some frames and the PLOTS script then performs analysis of 
# #pupil and motor positions and generates a PDF with results.
#python calibration/static_stability_analysis.py
#python calibration/static_stability_analysis_PLOTS.py --subdirectories "06-12-2024" "07-12-2024"