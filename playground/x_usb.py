import serial.tools.list_ports
import os
import time

ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
    print("{}: {} [{}]".format(port, desc, hwid))


device_to_cycle = "/dev/ttyACM2"  # Change this to your actual device path


def power_cycle_usb(device_path):
    """
    Power cycle a USB device by writing to the device files in /sys/bus/usb/devices/

    Parameters:
    -----------
    device_path: str
        The path to the USB device (e.g., /sys/bus/usb/devices/usb1)
    """
    try:
        # Turn off the USB device
        with open(os.path.join(device_path, "authorized"), "w") as f:
            f.write("0")
        time.sleep(1)  # Wait for 1 second

        # Turn on the USB device
        with open(os.path.join(device_path, "authorized"), "w") as f:
            f.write("1")
        print(f"Successfully power cycled the USB device at {device_path}")
    except Exception as e:
        print(f"Failed to power cycle the USB device at {device_path}: {e}")


# Find the corresponding sysfs path
usb_device_path = None
for port, desc, hwid in sorted(ports):
    if port == device_to_cycle:
        print(hwid)
        # Resolve the symlink to get the actual device path
        real_path = os.path.realpath(port)
        # Extract the relevant part of the path
        usb_device_path = os.path.dirname(real_path)
        break

if usb_device_path:
    print(f"Found device at {usb_device_path}")
    power_cycle_usb(usb_device_path)
else:
    print(f"Device {device_to_cycle} not found")
