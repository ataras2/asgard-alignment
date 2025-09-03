# *******************************************************************************
# E.S.O. - VLT project
#
#   agifbBackEnd.py
#
#  who       when        what
#  --------  ----------  ------------------------------------------------
#  smorel    2025-08-15  added HPOLn devices
#  smorel    2025-08-12  added BOTxn and SSFn devices
#  smorel    2024-10-29  created
#
# ******************************************************************************/
#
#  Example of ICS back-end for MCU.
#
# This python script shows how the ICS back-end server running on a
# Module Control Unit (MCU) of ASGARD shall respond to commands sent as
# JSON data by wag, including how it shall update the database on wag
# to have the device status reported by wag.
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
    def __init__(self, devName, semId, simu):
        self.name = devName
        self.semId = semId
        self.simulated = simu


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

# Array of semaphores (here, we consider using 60 semaphores, this number
# can be increased indeed)

sema = []
nbSemas = 70
for i in range(nbSemas):
    sema.append(0)

# Instanciate devices
# ...................................
# Update below the list of devices controlled by MCU
# (in this example, all the devices of Heimdallr, Baldr and Solarstein that
# are controlled by the "mimir" MCU are defined).
# Devices (axes) that cannot be moved in parallel (because they share the
# same controller) must have the same semaphore ID (= second parameter)
# ...................................

d = []
# Example: we cannot move HTPP1 and HTTP1 at the same time because they
# have the same semaphore ID (= 5)
d.append(device("HFO1", 1, 1))
d.append(device("HFO2", 2, 1))
d.append(device("HFO3", 3, 1))
d.append(device("HFO4", 4, 1))
d.append(device("HTPP1", 5, 1))
d.append(device("HTPP2", 6, 1))
d.append(device("HTPP3", 7, 1))
d.append(device("HTPP4", 8, 1))
d.append(device("HTTP1", 5, 1))
d.append(device("HTTP2", 6, 1))
d.append(device("HTTP3", 7, 1))
d.append(device("HTTP4", 8, 1))
d.append(device("HTPI1", 9, 1))
d.append(device("HTPI2", 10, 1))
d.append(device("HTPI3", 11, 1))
d.append(device("HTPI4", 12, 1))
d.append(device("HTTI1", 9, 1))
d.append(device("HTTI2", 10, 1))
d.append(device("HTTI3", 11, 1))
d.append(device("HTTI4", 12, 1))
d.append(device("HDL1", 13, 1))
d.append(device("HDL2", 14, 1))
d.append(device("HDL3", 15, 1))
d.append(device("HDL4", 16, 1))
d.append(device("HPOL1", 58, 1))
d.append(device("HPOL2", 59, 1))
d.append(device("HPOL3", 60, 1))
d.append(device("HPOL4", 61, 1))
d.append(device("BTP1", 17, 1))
d.append(device("BTP2", 18, 1))
d.append(device("BTP3", 19, 1))
d.append(device("BTP4", 20, 1))
d.append(device("BTT1", 17, 1))
d.append(device("BTT2", 18, 1))
d.append(device("BTT3", 19, 1))
d.append(device("BTT4", 20, 1))
d.append(device("BFA1", 21, 1))
d.append(device("BFA2", 22, 1))
d.append(device("BFA3", 23, 1))
d.append(device("BFA4", 24, 1))
d.append(device("BSA1", 25, 1))
d.append(device("BSA2", 26, 1))
d.append(device("BSA3", 27, 1))
d.append(device("BSA4", 28, 1))
d.append(device("BDS1", 29, 1))
d.append(device("BDS2", 30, 1))
d.append(device("BDS3", 31, 1))
d.append(device("BDS4", 32, 1))
d.append(device("BMX1", 33, 1))
d.append(device("BMX2", 34, 1))
d.append(device("BMX3", 35, 1))
d.append(device("BMX4", 36, 1))
d.append(device("BMY1", 33, 1))
d.append(device("BMY2", 34, 1))
d.append(device("BMY3", 35, 1))
d.append(device("BMY4", 36, 1))
d.append(device("BLF1", 37, 1))
d.append(device("BLF2", 38, 1))
d.append(device("BLF3", 39, 1))
d.append(device("BLF4", 40, 1))
d.append(device("BFO", 41, 1))
d.append(device("BOTP2", 42, 1))
d.append(device("BOTT2", 42, 1))
d.append(device("BOTP3", 43, 1))
d.append(device("BOTT3", 43, 1))
d.append(device("BOTP4", 44, 1))
d.append(device("BOTT4", 44, 1))
d.append(device("BFO", 45, 1))
d.append(device("SSS", 46, 1))
d.append(device("SDLA", 47, 1))
d.append(device("SDL12", 48, 1))
d.append(device("SDL34", 49, 1))
d.append(device("SSF1", 50, 1))
d.append(device("SSF2", 51, 1))
d.append(device("SSF3", 52, 1))
d.append(device("SSF4", 53, 1))
d.append(device("SGL", 54, 1))
d.append(device("SRL", 55, 1))
d.append(device("SBB", 56, 1))
d.append(device("SSP", 57, 1))

# Total number of devices that can be controlled by the MCU
nbCtrlDevs = len(d)

# Template of reply in JSON format

reply = {"reply": {"content": "????", "time": "YYYY-MM-DDThh:mm:ss", "parameters": []}}

print("")
print("ICS back-end server, v. 2 ")
print("--------------------------")
context = zmq.Context()
# Create server socket (listening to ic0fbControl process on wag)
srvSocket = context.socket(zmq.REP)
srvSocket.bind("tcp://*:5556")
print("Created server socket")

running = 1
cntdwnSetup = 0

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
        command = json.loads(message)
        cmdName = command["command"]["name"]
        timeStampIn = command["command"]["time"]

        # Verification of received time-stamp (to do...)
        # If the timestamp is invalid, set cmdName to "none",
        # so no command will be processed but a reply will be sent
        # back to the client (set replyContent to "ERROR")

        ################################
        # Process the received command:
        ################################

        # Case of "setup" (sent by wag to move devices)

        if "setup" in cmdName:
            stopped = 0
            nbDevs = len(command["command"]["parameters"])
            # Free all semaphores
            for i in range(nbSemas):
                sema[i] = 0
            # Create a double-list of devices to move
            setupList = [[], []]
            for i in range(nbDevs):
                kwd = command["command"]["parameters"][i]["name"]
                val = command["command"]["parameters"][i]["value"]
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

            # Move devices (if two batches, move the first one)
            batch = 0
            if len(setupList[batch]) > 0:
                print("batch", batch, "of devices to move:")
                reply["reply"]["parameters"].clear()
                for s in setupList[batch]:
                    print("Moving: ", s.dev, "to: ", s.val, "( setting", s.mType, " )")

                    # ......................................................
                    # Add here call to controller-specific functions that
                    # move the device "s.dev" to the requested position
                    # "s.val", according to "s.mType"
                    # ......................................................

                    # Inform wag ICS that the device is moving
                    attribute = "<alias>" + s.dev + ":DATA.status0"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": "MOVING"}
                    )

            # Simulate setup progress by setting a countdown
            # (that will be decremented when a "poll" command is received)
            cntdwnSetup = 3

            # Once setup is forwarded to the devices, reply OK if everything is
            # normal. This means that the setup has started, no that it is done!

            reply["reply"]["content"] = "OK"

        # Case of "poll" (sent by wag to get the status of the
        # last setup sent. Normally, wag sends a "poll" every
        # second during a setup)

        elif "poll" in cmdName:
            # --------------------------------------------------
            # Add here call to query the status of the batch of
            # devices that is concerned by the last setup command
            # If they all reach the target position or if
            # a STOP command occured, set batchDone to 1
            #
            # In this example of back-end server, we simulate
            # that by checking the cntdwnSetup variable
            # --------------------------------------------------
            if (cntdwnSetup == 0) or (stopped == 1):
                batchDone = 1
            else:
                batchDone = 0

            reply["reply"]["parameters"].clear()
            if len(setupList[batch]) > 0:
                for s in setupList[batch]:
                    attribute = "<alias>" + s.dev + ":DATA.status0"
                    # Case of motor with named position requested
                    if s.mType == "NAME":
                        # If motor reached the position, we set the
                        # attribute to the target named position
                        # (given in the setup) otherwise we set it
                        # to MOVING
                        if batchDone == 1:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": s.val}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: normally the encoder position shall be
                        # reported along with the named position
                        # ...............................................
                        # => Call function to read the encoder position
                        #    store it in a variable "posEnc" and execute:
                        #
                        # attribute = "<alias>" + s.dev +":DATA.posEnc"
                        # dbMsg['command']['parameters'].\
                        # append({"attribute":attribute, "value":posEnc})

                    # Case of shutter or lamp
                    if s.mType == "ST":
                        # Here the device can be either a lamp or a shutter
                        # Add here code to find out the type of s.dev
                        # If it is a shutter do:
                        if batchDone == 1:
                            if s.val == "T":
                                reply["reply"]["parameters"].append(
                                    {"attribute": attribute, "value": "OPEN"}
                                )
                            else:
                                reply["reply"]["parameters"].append(
                                    {"attribute": attribute, "value": "CLOSED"}
                                )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # If it is a lamp, reuse the code above replacing
                        # OPEN  by ON and CLOSED by OFF

                    # Case of motor with absolute encoder position requested
                    if s.mType == "ENC":
                        if batchDone == 1:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: if motor is at limit, do:
                        # dbMsg['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                        # Report the absolute encoder position
                        # Here (simulation), we simply use the target
                        # position (even if the motor is supposed to move)
                        attribute = "<alias>" + s.dev + ":DATA.posEnc"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": s.val}
                        )
                        # Case of motor with relative encoder position
                        # not considered yet
                        # The simplest would be to read the encoder position
                        # and to update the database as for the previous case

            # Check if second batch remains to setup
            # (if no STOP command has been sent)
            if batchDone == 1:
                if (batch == 0) and (len(setupList[1]) > 0) and (stopped == 0):
                    batch = 1
                    print("batch", batch, "of devices to move:")
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
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    # Reset simulation of setup progress
                    cntdwnSetup = 3
                    reply["reply"]["content"] = "PENDING"
                else:
                    # All batches of setup are done
                    reply["reply"]["content"] = "DONE"
            else:
                cntdwnSetup = cntdwnSetup - 1
                reply["reply"]["content"] = "PENDING"

        # Case of sensor reading request

        elif "read" in cmdName:
            reply["reply"]["parameters"].clear()

            # -----------------------------------------------
            # Add here call to function to retrieve the
            # values (real numbers) of the sensors
            # ------------------------------------------------

            # Here is an example (simulation), in which we read
            # four sensor values that are put into the reply
            # one after the other. The ICS front end on wag
            # has to know the order to assign the right FITS LOG
            # keyword to each received value

            reply["reply"]["parameters"].append({"value": 11.11})
            reply["reply"]["parameters"].append({"value": 22.22})
            reply["reply"]["parameters"].append({"value": 33.33})
            reply["reply"]["parameters"].append({"value": 44.44})

            reply["reply"]["content"] = "OK"

        # Case of other commands. The parameters are either a list
        # of devices, or "all" to apply the command to all the devices

        else:

            reply["reply"]["parameters"].clear()
            nbDevs = len(command["command"]["parameters"])
            allDevs = False
            # Check if command applies to all the existing devices
            if (nbDevs == 1) and (
                command["command"]["parameters"][0]["device"] == "all"
            ):
                nbDevs = nbCtrlDevs
                allDevs = True

            for i in range(nbDevs):
                if allDevs:
                    dev = d[i].name
                else:
                    dev = command["command"]["parameters"][i]["device"].upper()

                if cmdName == "disable":
                    print("Power off device:", dev)

                    # ......................................................
                    # Add here call to power-off the device dev
                    # ......................................................

                elif cmdName == "enable":
                    print("Power on device:", dev)

                    # ......................................................
                    # Add here call to power-on the device dev
                    # ......................................................

                elif cmdName == "off":
                    print("Turning off device:", dev)
                    # .........................................................
                    # If needed, call controller-specific functions to power
                    # down the device. It may require initialization
                    # after a power up
                    # .........................................................

                    # Update the wagics database to show that the device is
                    # in LOADED state (value of "state" attribute has to be
                    # set to 3)

                    attribute = "<alias>" + dev + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif cmdName == "online":
                    print("Setting ONLINE device:", dev)
                    # .........................................................
                    # If needed, call controller-specific functions to
                    # have the devices ready for operations (power them up
                    # if they have not already been ipowered up by a STANDBY
                    # command) and initialize them (if required).
                    # .........................................................

                    # Update the wagics database to show that the device is
                    # in ONLINE state (value of "state" attribute has to be
                    # set to 3)

                    attribute = "<alias>" + dev + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 3}
                    )

                elif cmdName == "simulat":
                    print("Simulation of device", dev)
                    # Set the simulation flag of dev to 1
                    for devIdx in range(nbCtrlDevs):
                        if d[devIdx].name == dev:
                            break
                    d[devIdx].simulated = 1

                    # Update the wagics database  to show that the device
                    # is in simulation and is in LOADED state

                    attribute = "<alias>" + dev + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )
                    attribute = "<alias>" + dev + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif cmdName == "standby":
                    print("Setting STANDBY device:", dev)
                    # .........................................................
                    # If needed, call controller-specific functions to bring
                    # the device to a "parking" position and to power them
                    # off (they should not require initialization when
                    # going ONLINE again). This command is called at
                    # end-of-night instrument shutdown
                    # .........................................................

                    # Update the wagics database to show that the device is
                    # in STANDBY state (value of "state" attrivute has to be
                    # set to 2)

                    attribute = "<alias>" + dev + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 2}
                    )

                elif cmdName == "stop":
                    print("Stop device:", dev)

                    # ......................................................
                    # Add here call to immediately stop the motion of the
                    # device dev
                    # ......................................................

                    # If setup is in progress, consider it done

                    # Update of the device status (positions, etc...) will be
                    # done by the next "poll" command sent by wag

                elif cmdName == "stopsim":
                    print("Normal mode for device", dev)
                    # Set the simulation flag of dev to 0
                    for devIdx in range(nbCtrlDevs):
                        if d[devIdx].name == dev:
                            break
                    d[devIdx].simulated = 0

                    # Update the wagics database  to show that the device
                    # is not in simulation and is in LOADED state
                    # (it may require an initialization when going to
                    # ONLINE state)

                    attribute = "<alias>" + dev + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 0}
                    )
                    attribute = "<alias>" + dev + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

            if cmdName == "stop":
                stopped = 1

            reply["reply"]["content"] = "OK"

        # Send back reply to ic0fb process (wag)

        timeNow = datetime.datetime.now()
        timeStamp = timeNow.strftime("%Y-%m-%dT%H:%M:%S")
        reply["reply"]["time"] = timeStamp

        # Convert reply JSON structure into a character string
        # terminated with null character (because ic0fb process on wag
        # in coded in C++ and needs null character to mark end of the string)

        repMsg = json.dumps(reply) + "\0"
        print(repMsg)
        srvSocket.send_string(repMsg)

    except Exception as e:
        print(str(e))
        print("closing socket...")
        srvSocket.close()
        context.destroy()
        exit()
