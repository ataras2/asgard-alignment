# a temperature watch dog that uses the controllino, polls it and saves some data
# monitoring only, no PI setting\
# to be run from the base directory of the asgard_alignment package

import time
import asgard_alignment.controllino as co
import os

duration = 20 * 60  # seconds
sampling = 5  # seconds

cur_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
savepth = os.path.join("data", "templogs", f"tempWD_{cur_datetime}.log")

cc = co.Controllino("192.168.100.10", 23, init_motors=False)

temp_probes = [
    "Lower T",
    "Upper T",
    "Bench T",
    "Floor T",
]

start_time = time.time()
with open(savepth, "w") as f:
    f.write("Time," + ",".join(temp_probes) + "\n")
    while time.time() - start_time < duration:
        temps = []
        for probe in temp_probes:
            try:
                temp = cc.analog_input(probe)
                temps.append(temp)
            except Exception as e:
                print(f"Error getting temperature for {probe}: {e}")
                temps.append(None)
        if temps is None:
            print("Failed to get temperatures, retrying...")
            time.sleep(sampling)
            continue

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write(current_time + "," + ",".join(f"{temp:.2f}" for temp in temps) + "\n")
        print(f"{current_time}: {temps}")

        time.sleep(sampling)
