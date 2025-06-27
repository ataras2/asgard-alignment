import subprocess
import time

def run_scripts():
    try:
        # Run the first script with "flat" DM map
        subprocess.run(
            [
                "python",
                "/home/asg/Progs/repos/asgard-alignment/calibration/static_stability_analysis.py",
                "--cam_gain", "5",
                "--cam_fps", "2000",
                "--dm_map", "flat"
            ],
            check=True
        )
        print("Flat map script completed successfully.")
l
        time.sleep( 5 )

        # Run the second script with "cross" DM map
        subprocess.run(
            [
                "python",
                "/home/asg/Progs/repos/asgard-alignment/calibration/static_stability_analysis.py",
                "--cam_gain", "5",
                "--cam_fps", "2000",
                "--dm_map", "cross"
            ],
            check=True
        )
        print("Cross map script completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error: Script failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

# Run scripts every 3 hours
run_count = 1
while True:
    print(f'========= RUN {run_count} ========')
    run_scripts()
    run_count += 1
    time.sleep(10 * 60) # Sleep for 2 hours
