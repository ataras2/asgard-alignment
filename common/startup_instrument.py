import os
from asgard_alignment.PDU_telnet import AtenEcoPDU
import sys
import zmq
import asgard_alignment.controllino as co
import time


# Box power ON through 192.168.100.11, port [05]
pdu = AtenEcoPDU("192.168.100.11")
pdu.connect()
print("Powering on the box...")
pdu.switch_outlet_status(5, "on")
time.sleep(5)  # wait for the box to power on
# check
res = pdu.read_outlet_status(5)
if res != "on":
    print("Failed to power on the box. Exiting.")
    sys.exit(1)
print("Box powered on successfully.")


# Ping test 192.168.100.10
def ping_test(ip_address):
    response = os.system(f"ping -c 1 {ip_address} > /dev/null 2>&1")
    return response == 0


if not ping_test("192.168.100.10"):
    print("Ping test failed for 192.168.100.10 (controllino). Exiting.")
    sys.exit(1)

# run mds and engineering GUI
cmds = [
    "test_mds",
    "test_eng_gui",
]
for cmd in cmds:
    print(f"Running command: {cmd}")
    res = os.system(cmd)
    if res != 0:
        print(f"Command '{cmd}' failed. Exiting.")
        sys.exit(1)

print("All commands executed successfully. Instrument startup complete.")
print("Load a state using the gui, and run 'fetch' on the camera server")
