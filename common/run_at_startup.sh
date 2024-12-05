#!/bin/bash
# THIS IS NECESSARY TO RUN AT STARTUP TO LOAD THE FTDI DRIVER AND LET THE TIP/TILT MIRRORS BE RECOGNIZED BY THE SYSTEM
# >>>>>>>>A CRONJOB IS SET TO RUN THIS SCRIPT AT STARTUP!! <<<<<<<<<<<
#IF YOU WANT TO MOVE OR EDIT THIS SCRIPT UPDATE CRONJOB AS WELL!!!

# Load the ftdi_sio module with the specified vendor and product IDs
sudo modprobe ftdi_sio vendor=0x104d:3008 product=3008

# Add the vendor/product ID to the USB serial driver
sudo sh -c 'echo "104d 3008" > /sys/bus/usb-serial/drivers/ftdi_sio/new_id'

