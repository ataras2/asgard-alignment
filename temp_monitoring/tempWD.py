# a temperature watch dog that uses the controllino, polls it and saves some data
# monitoring only, no PI setting\
# to be run from the base directory of the asgard_alignment package

import time
import asgard_alignment.controllino as co
import os

duration = 1.5 * 60 * 60  # seconds
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

# control loop info
PI_infos_of_interest = [
    "m_pin_val",
    "setpoint",
    "integral",
    "k_prop",
    "k_int",
]

servo_names = ["Lower", "Upper"]

start_time = time.time()
with open(savepth, "w") as f:
    # temp probes first, followed by PI infos prefixed by servo name
    f.write(
        "Time,"
        + ",".join(temp_probes)
        + ","
        + ",".join(
            [f"{servo} {key}" for servo in servo_names for key in PI_infos_of_interest]
        )
        + "\n"
    )
    try:
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

            PI_infos = []
            for i, servo in enumerate(servo_names):
                try:
                    info = cc.read_PI_loop_info(servo)
                    PI_infos.append(info)
                except Exception as e:
                    print(f"Error getting PI info for {servo}: {e}")
                    PI_infos.append(None)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(current_time + "," + ",".join(f"{temp:.2f}" for temp in temps))

            # write out the PI infos, prefixed by the servo name
            for i, info in enumerate(PI_infos):
                if info is not None:
                    f.write(
                        ","
                        + ",".join(
                            (
                                f"{info.get(key, 'None'):.2f}"
                                if isinstance(info.get(key), (int, float))
                                else str(info.get(key))
                            )
                            for key in PI_infos_of_interest
                        )
                    )

            f.write("\n")
            f.flush()
            print(f"{current_time}: {temps}")

            time.sleep(sampling)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Exiting gracefully and saving log.")
