# *******************************************************************************
# E.S.O. - VLT project
#
#   agifbBackEnd.py
#
#  who       when        what
#  --------  ----------  ------------------------------------------------
#  smorel    2024-10-29  created
#
# ******************************************************************************/
#
#  Example of ICS back-end for MCU.
#
# ******************************************************************************/

import time
import datetime
import math
import zmq
import re
import json


####################################################
# Main function
####################################################

# Device class (each instance of this class describes a device that is
# controlled by the MCU)


class device:
    def __init__(self, devName, semId):
        self.name = devName
        self.semId = semId


# Setup class (each instance of this class describes an elementary setup
# for one device)


class setup:
    def __init__(self, dev, mType, val):
        self.dev = dev
        self.mType = mType
        self.val = val


#
# "Main" starts here
#

# Array of semaphores (here, we consider using 16 semaphores, this number
# can be increased indeed)

sema = []
nbSemas = 16
for i in range(nbSemas):
    sema.append(0)

# Instanciate devices
# ...................................
# Update below the list of devices controlled by MCU
# (in this example, devices from various modules are used, in
# order to test a template, with only one instance of this script
# running locally).
# Devices (axes) that cannot be moved in parallel (because they share the
# same controller) must have the same semaphore ID (= second parameter)
# ...................................

d = []
# Example: we cannot move HTPP1 and HTTP1 at the same time because they
# have the same semaphore ID (= 1)
d.append(device("HTPP1", 1))
d.append(device("HTTP1", 1))
d.append(device("SSS", 2))
d.append(device("SSF", 3))
d.append(device("SGL", 4))
d.append(device("SRL", 5))
d.append(device("SBB", 6))
d.append(device("SSP", 7))
d.append(device("NSH1", 7))
d.append(device("NSH2", 8))
d.append(device("NSH3", 9))
d.append(device("NSH4", 10))
d.append(device("FPSH1", 11))
d.append(device("FPSH2", 12))
d.append(device("FPSH3", 13))
d.append(device("FPSH4", 14))

# Total number of devices that can be controlled by the LCU
nbCtrlDevs = len(d)

# Template of message in JSON to send,  in order to update the database on wag

dbMsg = {
    "command": {
        "name": "write",
        "time": "YYYY-MM-DDThh:mm:ss",
        "parameters": [],
    }
}

context = zmq.Context()
# Create server socket (listening to ic0fbControl process on wag)
srvSocket = context.socket(zmq.REP)
srvSocket.bind("tcp://*:5556")
print("Created server socket")
# Create client socket (sending database update requests to agifbDbRelay
# process on wag)
cliSocket = context.socket(zmq.PUSH)
cliSocket.connect("tcp://localhost:5561")

running = 1

# Main loop

while running == 1:

    # Listen on ZMQ server socket, parse message

    print("Listening to client...")
    try:
        message = srvSocket.recv_string()
        print("Received message :")

        # Need to remove the last character (probably \0, because message
        # is received from a process written in C++), otherwise call to
        # "json.loads" fails

        message = message.rstrip(message[-1])
        print(message)
        json_data = json.loads(message)
        cmdName = json_data["command"]["name"]
        timeStampIn = json_data["command"]["time"]

        # Verification of received time-stamp (to do...)
        # If the timestamp is invalid, set cmdName to "none",
        # so no command will be processed but a reply will be sent
        # back to the client (set replyContent to "ERROR")

        ################################
        # Process the received command:
        ################################

        # Case of "online" (sent by wag when bringing ICS online, to check
        # that MCUs are alive and ready)

        if "online" in cmdName:

            # .............................................................
            # If needed, call controller-specific functions to power up
            # the devices and have them ready for operations
            # .............................................................

            # Update the wagics database to show all the devices in ONLINE
            # state (value of "state" attribute has to be set to 3)

            dbMsg["command"]["parameters"].clear()
            for i in range(nbCtrlDevs):
                attribute = "<alias>" + d[i].name + ".state"
                dbMsg["command"]["parameters"].append(
                    {"attribute": attribute, "value": 3}
                )

            # Send message to wag to update the database
            timeNow = datetime.datetime.now()
            timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
            dbMsg["command"]["time"] = timeStamp
            outputMsg = json.dumps(dbMsg) + "\0"

            cliSocket.send_string(outputMsg)
            print(outputMsg)

            replyContent = "OK"

        # Case of "standby" (sent by wag when bringing ICS standby,
        # usually when the instrument night operations are finished)

        if "standby" in cmdName:

            # .............................................................
            # If needed, call controller-specific functions to bring some
            # devices to a "parking" position and to power them off
            # .............................................................

            # Update the wagics database to show all the devices in STANDBY
            # state (value of "state" attrivute has to be set to 2)

            dbMsg["command"]["parameters"].clear()
            for i in range(nbCtrlDevs):
                attribute = "<alias>" + d[i].name + ".state"
                dbMsg["command"]["parameters"].append(
                    {"attribute": attribute, "value": 2}
                )

            # Send message to wag to update the database
            timeNow = datetime.datetime.now()
            timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
            dbMsg["command"]["time"] = timeStamp
            outputMsg = json.dumps(dbMsg) + "\0"

            cliSocket.send_string(outputMsg)
            print(outputMsg)

            replyContent = "OK"

        # Case of "setup" (sent by wag to move devices)

        if "setup" in cmdName:
            nbDevs = len(json_data["command"]["parameters"])
            # Free all semaphores
            for i in range(nbSemas):
                sema[i] = 0
            # Create a double-list of devices to move
            setupList = [[], []]
            for i in range(nbDevs):
                kwd = json_data["command"]["parameters"][i]["name"]
                val = json_data["command"]["parameters"][i]["value"]
                print("Setup:", kwd, "to", val)

                # Keywords are in the format: INS.<device>.<motion type>

                prefixes = kwd.split(".")
                dev = prefixes[1]
                mType = prefixes[2]
                print("Device:", dev, " - motion type:", mType)

                # mType can be one of these words:
                # NAME   = Named position (e.g., IN, OUT, J1, H3, ...)
                # ENC    = Absolute encoder position
                # ENCREL = Relative encoder postion (can be negative)
                # ST     = State. Given value is equal to either T or F.
                #          if device is shutter: T = open, F = closed.
                #          if device is lamp: T = on, F = off.

                # Look if device exists in list
                # (something should be done if device does not exist)
                for devIdx in range(nbCtrlDevs):
                    if d[devIdx].name == dev:
                        break

                semId = d[devIdx].semId
                if sema[semId] == 0:
                    # Semaphore is free =>
                    # Device can be moved now
                    setupList[0].append(setup(dev, mType, val))
                    sema[semId] = 1
                else:
                    # Semaphore is already taken =>
                    # Device will be moved in a second batch
                    setupList[1].append(setup(dev, mType, val))

            # Move devices (two batchesi if needed)
            for batch in range(2):
                if len(setupList[batch]) > 0:
                    print("batch", batch, "of devices to move:")
                    dbMsg["command"]["parameters"].clear()
                    for s in setupList[batch]:
                        print(
                            "Moving: ", s.dev, "to: ", s.val, "( setting", s.mType, " )"
                        )

                        # ......................................................
                        # Add here call to controller-specific functions that
                        # move the device "s.dev" to the requested position
                        # "s.val", according to "s.mType"
                        # ......................................................

                        # Inform wag ICS that the device is moving
                        attribute = "<alias>" + s.dev + ":DATA.status0"
                        dbMsg["command"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    # Send message to wag to update the database
                    timeNow = datetime.datetime.now()
                    timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
                    dbMsg["command"]["time"] = timeStamp
                    outputMsg = json.dumps(dbMsg) + "\0"

                    cliSocket.send_string(outputMsg)
                    print(outputMsg)

                    # Simulates a motion delay (just for test, to be deleted)
                    time.sleep(1)

                    # ........................................................
                    # Add here calls to read (every 1 to 3 seconds) the position
                    # of the devices and update the database of wag (using the
                    # code below to generate the JSON message)
                    # ........................................................

                    # ........................................................
                    # Add here call to check that devices have reached their
                    # requested positions. Once done, inform wag as follows:
                    # ........................................................

                    dbMsg["command"]["parameters"].clear()
                    for s in setupList[batch]:
                        attribute = "<alias>" + s.dev + ":DATA.status0"
                        # Case of motor with named position requested
                        if s.mType == "NAME":
                            dbMsg["command"]["parameters"].append(
                                {"attribute": attribute, "value": s.val}
                            )
                            # Note: normally the encoder position shall be
                            # reported along with the named position
                            # ...............................................
                            # => Call function to read the encoder position
                            #    store it in a variable "posEnc" and execute:
                            #
                            # attribute = "<alias>" + s.dev +":DATA.posEnc"
                            # dbMsg['command']['parameters'].append({"attribute":attribute, "value":posEnc})

                        # Case of shutter or lamp
                        if s.mType == "ST":
                            # Here the device can be either a lamp or a shutter
                            # Add here code to find out the type of s.dev
                            # If it is a shutter do:
                            if s.val == "T":
                                dbMsg["command"]["parameters"].append(
                                    {"attribute": attribute, "value": "OPEN"}
                                )
                            else:
                                dbMsg["command"]["parameters"].append(
                                    {"attribute": attribute, "value": "CLOSED"}
                                )
                        # If it is a lamp, reuse the code above replacing
                        # OPEN  by ON and CLOSED by OFF

                        # Case of motor with absolute encoder position requested
                        if s.mType == "ENC":
                            dbMsg["command"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                            # Note: if motor is at limit, do:
                            # dbMsg['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                            attribute = "<alias>" + s.dev + ":DATA.posEnc"
                            dbMsg["command"]["parameters"].append(
                                {"attribute": attribute, "value": s.val}
                            )
                        # Case of motor with relative encoder position
                        # not considered yet
                        # The simplest would be to read the encoder position
                        # and to update the database as for the previous case

                    # Send message to wag to update its database
                    timeNow = datetime.datetime.now()
                    timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
                    dbMsg["command"]["time"] = timeStamp
                    outputMsg = json.dumps(dbMsg) + "\0"

                    cliSocket.send_string(outputMsg)
                    print(outputMsg)

            # Once setup is completed, reply OK if everything is
            # normal.

            replyContent = "OK"

        # Case of "stop" (sent by wag to immediately stop the devices)

        if "stop" in cmdName:
            nbDevs = len(json_data["command"]["parameters"])
            for i in range(nbDevs):
                dev = json_data["command"]["parameters"][i]["device"]
                print("Stop device:", dev)

                # ......................................................
                # Add here call to stop the motion of the device dev
                # ......................................................

            replyContent = "OK"

        # Case of "disable" (sent by wag to power-off devices)

        if "disable" in cmdName:
            nbDevs = len(json_data["command"]["parameters"])
            for i in range(nbDevs):
                dev = json_data["command"]["parameters"][i]["device"]
                print("Power off device:", dev)

                # ......................................................
                # Add here call to power-off the device dev
                # ......................................................

            replyContent = "OK"

        # Case of "enable" (sent by wag to power-on devices)

        if "enable" in cmdName:
            nbDevs = len(json_data["command"]["parameters"])
            for i in range(nbDevs):
                dev = json_data["command"]["parameters"][i]["device"]
                print("Power on device:", dev)

                # ......................................................
                # Add here call to power-on the device dev
                # ......................................................

            replyContent = "OK"

        # Send back reply to ic0fb process

        timeNow = datetime.datetime.now()
        timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
        reply = (
            '{\n\t"reply" :\n\t{\n\t\t"content" : "'
            + replyContent
            + '",\n\t\t"time" : "'
            + timeStamp
            + '"\n\t}\n}\n\0'
        )
        print(reply)
        srvSocket.send_string(reply)

    except Exception as e:
        print(str(e))
        print("closing socket...")
        srvSocket.close()
        context.destroy()
        exit()
