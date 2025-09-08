import os
from asgard_alignment.PDU_telnet import AtenEcoPDU
import sys
import zmq
import asgard_alignment.controllino as co
import time

LOWER_BOX_OUTLET = 5
C_RED_OUTLET = 6


def ping_device(ip):
    response = os.system(f"ping -c 1 {ip} > /dev/null 2>&1")
    return response == 0


def open_zmq_connection(port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 10000)
    server_address = f"tcp://192.168.100.2:{port}"
    socket.connect(server_address)
    return socket


def send_and_get_response(socket, string):
    socket.send_string(string)
    res = socket.recv_string()
    return res


def shutdown(inc_CRED):
    mds_connection = open_zmq_connection(5555)
    date = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        res = send_and_get_response(mds_connection, f"save all before_shutdown_{date}")
        print("saved", res)
    except zmq.error.Again:
        inp = input(
            "MDS did not respond (and hence state is not saved). Do you want to continue with shutdown? (y/n): "
        )
        if inp.lower() != "y":
            print("Aborting shutdown.")
            return
        print("Proceeding with shutdown...")

    if inc_CRED:
        c_red_connection = open_zmq_connection(6667)

    cc = co.Controllino("192.168.100.10", init_motors=False)

    # turn off all sources: SRL, SGL and SBB
    lamps = ["SRL", "SGL", "SBB"]

    for lamp in lamps:
        send_and_get_response(mds_connection, f"off {lamp}")
        time.sleep(1)  # wait for the command to be processed

    # flippers up
    names = [f"SSF{i}" for i in range(1, 5)]
    for i, flipper in enumerate(names):
        message = f"moveabs {flipper} 1.0"
        send_and_get_response(mds_connection, message)
        time.sleep(2)  # wait for the command to be processed

    pdu = AtenEcoPDU("192.168.100.11")
    pdu.connect()
    pre_shutdown_current = float(pdu.read_power_value("olt", LOWER_BOX_OUTLET, "curr"))
    print(f"Pre-shutdown current: {pre_shutdown_current} A")

    devices = [
        "USB upper power",
        "X-MCC (BMX,BMY)",
        "X-MCC (BFO,SDL,BDS,SSS)",
        "LS16P (HFO)",
        "DM1",
        "DM2",
        "DM3",
        "DM4",
        "USB hubs",
    ]
    for device in devices:
        cc.turn_off(device)
    print("Waiting for all devices to turn off...")

    time.sleep(5)  # wait for devices to turn off

    post_shutdown_current = float(pdu.read_power_value("olt", LOWER_BOX_OUTLET, "curr"))
    print(f"Post-shutdown current: {post_shutdown_current} A")

    input("Close the MDS and engineering GUI, then press Enter to continue...")

    if not inc_CRED:
        input(
            "In C red server terminal, type 'stop' and then type 'exit'. Then press Enter here to continue..."
        )

    if inc_CRED:
        print("Closing C RED...")
        send_and_get_response(c_red_connection, "stop")
        send_and_get_response(c_red_connection, 'cli "set cooling off"')
        send_and_get_response(c_red_connection, 'cli "shutdown"')
        print("C RED shutdown command sent.")

        pdu.switch_outlet_status(C_RED_OUTLET, "off")
        time.sleep(7)
        res = pdu.read_outlet_status(C_RED_OUTLET)
        if res == "off":
            print("C RED outlet is off.")

    print("Turning off PDU outlet...")
    pdu.switch_outlet_status(LOWER_BOX_OUTLET, "off")
    time.sleep(7)  # wait for PDU to turn off

    res = pdu.read_outlet_status(LOWER_BOX_OUTLET)
    if res == "off":
        print("PDU outlet is off.")
    else:
        print(f"PDU outlet status: {res}. Please check the PDU manually.")

    # Verify no response from ping 192.168.100.111, 192.168.100.10
    ips_to_check = ["192.168.100.111", "192.168.100.10"]

    for ip in ips_to_check:
        if ping_device(ip):
            print(f"Device {ip} is still reachable. Please check manually.")
        else:
            print(f"Device {ip} is not reachable, as expected.")

    print("Shutdown procedure completed successfully.")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "inc_CRED":
        inc_CRED = True
    else:
        inc_CRED = False

    shutdown(inc_CRED)
