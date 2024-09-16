import asgard_alignment.Instrument


instr = asgard_alignment.MultiDeviceServer.MultiDeviceServer(
    "motor_info_sydney_subset.json"
)

print("HFO1 in instrument?", "HFO1" in instr._motors)
