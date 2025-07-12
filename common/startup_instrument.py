"""
A full system startup, starting from the case where only mimir is on
"""

import os
from asgard_alignment.PDU_telnet import AtenEcoPDU
import sys
import time


def ping_test(ip_address):
    response = os.system(f"ping -c 1 {ip_address} > /dev/null 2>&1")
    return response == 0


def power_on_all():
    # Box power ON through 192.168.100.11, port [05]
    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()
    print("Powering on the camera...")
    pdu.switch_outlet_status(6, "on")

    print("Powering on the box...")
    pdu.switch_outlet_status(5, "on")

    is_on = {
        5: False,
        6: False,
    }

    while not all(is_on.values()):
        for outlet in is_on.keys():
            res = pdu.read_outlet_status(outlet)
            if res == "on":
                is_on[outlet] = True
            else:
                print(f"Outlet {outlet} status: {res}")
        if not all(is_on.values()):
            print("Waiting for the power on...")
            time.sleep(1)  # wait for the box to power on

    print("Box and camera powered on successfully.")

    # Ping test 192.168.100.10
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


def power_on_instrument_only():

    # Box power ON through 192.168.100.11, port [05]
    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()
    print("Powering on the box...")
    pdu.switch_outlet_status(5, "on")
    time.sleep(4)  # wait for the box to power on
    res = pdu.read_outlet_status(5)
    time.sleep(4)  # wait for the box to power on
    # check
    res = pdu.read_outlet_status(5)
    if res != "on":
        print(res)
        print("Failed to power on the box. Exiting.")
        sys.exit(1)
    print("Box powered on successfully.")

    # Ping test 192.168.100.10
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


def main():
    # check which mode to run
    inp = (
        input("Do you want to power on the C RED one camera too? (y/n): ")
        .strip()
        .lower()
    )
    if inp == "y":
        power_on_all()
    elif inp == "n":
        power_on_instrument_only()
    else:
        print("Invalid input. Please enter 'y' or 'n'.")
        sys.exit(1)
