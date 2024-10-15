# *******************************************************************************
# E.S.O. - VLT project
#
#   agifbMultiDevServer.py
#
#  who       when        what
#  --------  ----------  ------------------------------------------------
#  smorel    2024-09-17  created
#
# ******************************************************************************/

##########################################################################
# Adaptation in Python of agifbMultiDevServer.h and agifbMultiDevServer.C
#
# Note: "print" commands have been added just for debugging
##########################################################################

from datetime import datetime
import math
import zmq
import re

# Types of devices
MOTOR, SHUTTER, LAMP, SENSOR, IODEV = range(5)

# Set of devices
HFO1, HFO2, HFO3, HFO4, NSH1, SGL, SRL, SBB, SSP, NSENS = range(10)

# Codes for motor status
MOT_ERROR, MOT_STANDING, MOT_MOVING, MOT_INITIALISING = range(4)

# Motor commands
(
    MOT_NONE,
    MOT_INITIALISE,
    MOT_SETPOS,
    MOT_MOVEABS,
    MOT_MOVEREL,
    MOT_MOVEVEL,
    MOT_NEWVEL,
    MOT_NEWPOS,
) = range(8)

# Motor states
IDLE, RESET_AXIS, MOVE, STOP, INIT = range(5)

# Codes for shutter status
SHT_ISCLOSED, SHT_ISOPEN, SHT_UNKNOWN, SHT_ERROR = range(4)

# Shutter commands
SHT_NONE, SHT_INITIALISE, SHT_CLOSE, SHT_OPEN = range(4)

# Shutter error codes
(
    SHT_OK,
    SHT_FAULT_SIG,
    SHT_NOT_IN_OP_STATE,
    SHT_NO_SIG_USED,
    SHT_BOTH_SIG_ACTIVE,
    SHT_LOCAL_MODE,
) = range(6)
SHT_NOT_INITIALISED = 10

# Codes for lamp status
LMP_ISOFF, LMP_WARMING_UP, LMP_ISON, LMP_COOLING_DOWN, LMP_ERROR = range(5)

# Lamp commands
LMP_NONE, LMP_INITIALISE, LMP_OFF, LMP_ON = range(4)

# Lamp error codes
(
    LMP_OK,
    LMP_FAULT_SIG,
    LMP_NOT_IN_OP_STATE,
    LMP_MAXON_TIMEOUT,
    LMP_NOT_COOLED,
    LMP_LOCAL_MODE,
    LMP_WHILE_ON_WENT_OFF,
    LMP_NO_FEEDBACK_SIG,
) = range(8)
LMP_NOT_INITIALISED = 10

# Codes for sensor status
SNS_READY, SNS_ERROR = range(2)

# Sensor commands
SNS_NONE, SNS_INITIALISE, SNS_ACTIVATE = range(3)

# Sensor error codes
SNS_OK = 0
SNS_NOT_IN_OP_STATE = 4


class device:
    def __init__(self, name, devType):

        # *******************************************************
        # COMPLETE THIS LIST OF PARAMETERS WITH WHAT IS REQUIRED
        # FOR THE LOW-LEVEL CONTROL OF DEVICES
        # (e.g.: device driver name, etc...)
        # *******************************************************

        # "public" parameters
        self.name = name

        if devType == MOTOR:
            self.nCommand = MOT_NONE
        if devType == LAMP:
            self.nCommand = LMP_NONE
        self.bExecute = False

        # control parameters (motors)
        self.lrPosition = 0.0
        self.lrVelocity = 0.0
        self.bStop = False
        self.bDisable = False
        self.bEnable = False
        self.bResetError = False

        # status parameters (all devices)
        self.bLocal = False
        self.bInitialised = False
        if devType == MOTOR:
            self.nLastCommand = MOT_NONE
            self.nErrorCode = 0
        elif devType == LAMP:
            self.nLastCommand = LMP_NONE
            self.nErrorCode = LMP_NOT_INITIALISED
        self.nErrorCode = 0

        # status parameters (motors)
        self.lrScaleFactor = 1.0
        self.lrPosTarget = 10.0
        self.lrPosActual = 0.0
        self.lrBacklash = 0.0
        self.nAxisStatus = MOT_STANDING
        self.nBacklashStep = 2
        self.nInfoData1 = 0
        self.nInfoData2 = 0
        self.nInitStep = 0
        self.bEnabled = False
        self.lrMaxPositionValue = 100.0
        self.lrMinPositionValue = 0.0
        self.bAtMaxPosition = False
        self.bAtMinPosition = False
        self.bBrakeActive = False
        self.lrModuloFactor = 0.0
        self.signals_0__bActive = False
        self.signals_1__bActive = False
        self.signals_2__bActive = False
        self.signals_3__bActive = False
        self.signals_4__bActive = False
        self.signals_5__bActive = False
        self.bRefSwitchActive = False
        # status parameters (shutters)
        self.nStatus = 0

        # status parameters (lamps)
        self.lrIntensity = 100.0
        self.nTimeOff = 0  # time in seconds since lamp was turned off
        self.nTimeOn = 0  # time in seconds since lamp was turned on

        # config parameters (lamps)
        self.nCooldown = (
            5  # time in seconds to wait before turning lamp on again, can be 0
        )
        self.nWarmup = 5  # time in seconds to wait before lamp is fully on
        self.nMaxOn = 4000

        # status parameters (sensors)
        self.arr_AI = [0.0] * 8

        # internal parameters (used by the finite-state machines)
        self._devType = devType
        self._state = 0
        self._t_last = 0
        self._t_on = 0
        self._t_off = 0


# Update all parameters of a motor from the previous update of a parameter


def update_motor_param(dev):
    print("Updating parameters of motor device #", dev)
    if p[dev].bEnable == True:
        p[dev].bEnabled = True
        p[dev].bEnable = False
    if p[dev].bDisable == True:
        p[dev].bEnabled = False
        p[dev].bDisable = False
    #    if m.bEnabled == True and m.bInitialized == True:

    # ***********************************************
    # * ADD HERE CALL TO CHECK IF MOTOR IS AT LIMIT *
    # ***********************************************
    # If motor (device indexed by D) is at limit,
    #       enable and execute the following code:

    #  m.bEnabled = False
    #  m.nAxisStatus = MOT_ERROR;
    #  m.nErrorCode = MOT_ERROR_LIMIT;
    #  m._state = IDLE;

    if p[dev].bStop == True:
        p[dev].bStop = False
        # *********************************************
        # * ADD HERE CALL TO STOP MOTOR (PANIC STOP) *
        # *********************************************

        p[dev]._state = STOP

    if p[dev].bResetError == True:  # unclear in which case this can actually be sent...
        # Reset axis
        p[dev].bResetError = False
        p[dev].bStop = False
        p[dev].bInitialised = False
        p[dev].nErrorCode = 0
        p[dev]._state = RESET_AXIS

    if p[dev].bExecute == True:
        print("Executing command...")
        # Execute the last command received (as a parameter)
        p[dev].nLastCommand = p[dev].nCommand
        p[dev].bExecute = False
        p[dev].nCommand = MOT_NONE
        if p[dev].bEnabled == False or (
            p[dev].nLastCommand != MOT_INITIALISE and p[dev].bInitialised == False
        ):
            print("Error: motor not initialised")
            # Error: motor not initialised
            p[dev].nAxisStatus = MOT_ERROR
            p[dev].nErrorCode = 1000
            p[dev]._state = IDLE
        else:
            if p[dev].nLastCommand == MOT_INITIALISE:
                # Initialise motor
                p[dev].bInitialised = False
                p[dev].nErrorCode = 0
                p[dev].nAxisStatus = MOT_INITIALISING
                print("Initialising")

                # ***********************************************
                # * ADD HERE CALL TO START MOTOR INITIALISATION *
                # *
                # * If the motors do not need initialisation by
                # * the wag ICS, ignore this but keep the
                # * code that modifies the p[dev] parameters
                # ***********************************************
                p[dev]._state = INIT
            if p[dev].nLastCommand == MOT_MOVEABS:
                # Move motor to absolute position
                p[dev].nErrorCode = 0
                p[dev].nAxisStatus = MOT_MOVING
                p[dev].lrPosTarget = p[dev].lrPosition

                # *************************************************
                # ADD HERE CALL TO START MOTOR MOTION TO ABS ENC *
                # *************************************************
                p[dev]._state = MOVE

            if p[dev].nLastCommand == MOT_MOVEREL:
                # Move motor to relative position
                p[dev].nErrorCode = 0
                p[dev].nAxisStatus = MOT_MOVING
                # Convert relative move to absolute move
                p[dev].lrPosTarget = p[dev].lrPosition + p[dev].lrPosActual

                # *************************************************
                # ADD HERE CALL TO START MOTOR MOTION TO ABS ENC *
                # *************************************************
                p[dev]._state = MOVE
                # the following line is only used for the built-in SW simulation
                p[dev]._t_last = math.floor(datetime.timestamp(datetime.now()))


# Update all parameters of a lamp from the previous update of a parameter


def update_lamp_param(dev):
    print("Updating parameters of lamp device #", dev)
    if p[dev].bExecute == True:
        p[dev].nLastCommand = p[dev].nCommand
        p[dev].bExecute = False
        p[dev].nCommand = LMP_NONE
        if p[dev].nLastCommand == LMP_INITIALISE:
            if p[dev].nTimeOn > 0:
                # ***************************************************
                # ADD HERE CALL TO GET LAMP INITIALISED (if needed)
                # AND OFF
                # ***************************************************
                p[dev]._t_off = math.floor(datetime.timestamp(datetime.now()))
                p[dev].nStatus = LMP_COOLING_DOWN
            else:
                p[dev].nStatus = LMP_ISOFF
                p[dev].nErrorCode = LMP_OK
                p[dev].bInitialised = True
                p[dev]._t_off = math.floor(datetime.timestamp(datetime.now()))
        else:
            if p[dev].bInitialised == True:
                if p[dev].nLastCommand == LMP_OFF:
                    # ************************************
                    # ADD HERE CALL TO TURN LAMP OFF
                    # ************************************
                    print("Turning lamp off...")
                    p[dev]._t_off = math.floor(datetime.timestamp(datetime.now()))
                    p[dev].nTimeOn = 0
                    p[dev].nTimeOff = 0
                    p[dev].nErrorCode = LMP_OK
                    p[dev].nStatus = LMP_COOLING_DOWN
                if p[dev].nLastCommand == LMP_ON:
                    t_now = math.floor(datetime.timestamp(datetime.now()))
                    # Check lamp is cold before turning it on again
                    # (to prevent damaging for some lamps)
                    if (
                        p[dev].nStatus == LMP_COOLING_DOWN
                        or (t_now - p[dev]._t_off) < p[dev].nCooldown
                    ):
                        p[dev].nErrorCode = LMP_NOT_COOLED
                        p[dev].nStatus = LMP_ERROR
                    else:
                        # ********************************
                        # ADD HERE CALL TO TURN LAMP ON
                        # ********************************
                        print("Turning lamp on...")
                        p[dev]._t_on = math.floor(datetime.timestamp(datetime.now()))
                        p[dev].nTimeOn = 0
                        p[dev].nTimeOff = 0
                        p[dev].nErrorCode = LMP_OK
                        p[dev].nStatus = LMP_WARMING_UP


# Update finite-state machine of a motor device


def update_motor_fsm(dev):
    if p[dev]._state == IDLE:
        p[dev].bResetError = False
        p[dev].bStop = False

    elif p[dev]._state == RESET_AXIS:
        # ******************************************
        # * ADD HERE CALL TO CHECK STATUS OF RESET *
        # ******************************************
        #     If reset failed, execute the following:
        #     p[dev].nErrorCode = <the error code>
        #     p[dev].nAxisStatus = MOT_ERROR
        #     p[dev]._state = IDLE
        #
        #     If reset succeded, execute the following:

        p[dev].nErrorCode = 0
        p[dev].nAxisStatus = MOT_STANDING
        p[dev]._state = IDLE

    elif p[dev]._state == STOP:
        # *********************************************
        # * ADD HERE CALL TO CHECK STATUS OF STOPPING *
        # *********************************************
        #     If stop failed, execute the following:
        #     p[dev].nErrorCode = <the error code>
        #     p[dev].nAxisStatus = MOT_ERROR
        #     p[dev]._state = IDLE
        #
        #     If stop succeeded, execute the following:

        p[dev].nErrorCode = 0
        p[dev].nAxisStatus = MOT_STANDING
        p[dev]._state = IDLE

    elif p[dev]._state == INIT:
        # ***************************************************
        # * ADD HERE CALL TO CHECK STATUS OF INITIALISATION *
        # ***************************************************
        #     If initialisation failed, execute the following:
        #     p[dev].nErrorCode = <the error code>
        #     p[dev].nAxisStatus = MOT_ERROR
        #     p[dev].bInitialised = False
        #     p[dev]._state = IDLE;
        #
        #     If initialisation succeeded, execute the following:
        p[dev].nErrorCode = 0
        p[dev].nAxisStatus = MOT_STANDING
        p[dev].bInitialised = True
        p[dev]._state = IDLE

    elif p[dev]._state == MOVE:
        atPosition = False

        # ***********************************************
        # * ADD HERE CALL TO GET MOTOR ENCODER POSITION *
        # * and to update m.lrPosActual                   *
        # ***********************************************
        #     The following is a built-in SW simulation of motor motion,
        #     remove it for actual implementation */

        t_now = math.floor(datetime.timestamp(datetime.now()))
        if t_now > p[dev]._t_last:
            x_now = p[dev].lrPosActual
            x_targ = p[dev].lrPosTarget
            if math.fabs(x_now - x_targ) < 1.0:
                atPosition = True
            else:
                if x_now < x_targ:
                    x_now = x_now + 1
                else:
                    x_now = x_now - 1
                p[dev].lrPosActual = x_now
            p[dev]._t_last = t_now
        # end of built-in simulation of motor motion

        # *********************************************
        # * ADD HERE CALL TO CHECK STATUS OF MOTION   *
        # *********************************************
        #     If motion failed (and motor stopped), execute the following:
        #     p[dev].nErrorCode = <the error code>
        #     p[dev].nAxisStatus = MOT_ERROR
        #     p[dev]._state = IDLE
        #
        #     If motion succeeded, execute the following:

        if atPosition == True:
            p[dev].nErrorCode = 0
            p[dev].nAxisStatus = MOT_STANDING
            p[dev].bInitialised = True
            p[dev]._state = IDLE


# Update finite-state machine of a lamp device


def update_lamp_fsm(dev):
    t_now = math.floor(datetime.timestamp(datetime.now()))
    if p[dev].nStatus == LMP_WARMING_UP:
        p[dev].nTimeOn = t_now - p[dev]._t_on
        if p[dev].nTimeOn > p[dev].nWarmup:
            p[dev].nStatus = LMP_ISON
    elif p[dev].nStatus == LMP_ISON:
        p[dev].nTimeOn = t_now - p[dev]._t_on
        if p[dev].nTimeOn > p[dev].nMaxOn:
            # *******************************************
            # ADD HERE CALL TO TURN LAMP OFF FOR SAFETY
            # (lamp has been on for too long)
            # *******************************************
            p[dev]._t_off = math.floor(datetime.timestamp(datetime.now()))
            p[dev].nTimeOff = 0
            p[dev].nTimeOn = 0
            p[dev].nStatus = LMP_COOLING_DOWN
    elif p[dev].nStatus == LMP_COOLING_DOWN:
        p[dev].nTimeOff = t_now - p[dev]._t_off
        if p[dev].nTimeOff > p[dev].nCooldown:
            p[dev].nStatus = LMP_ISOFF
    if p[dev].nStatus == LMP_ISOFF:
        p[dev].nTimeOff = t_now - p[dev]._t_off
    print(
        "Lamp state is",
        p[dev].nStatus,
        " ; tOn =",
        p[dev].nTimeOn,
        " ; tOff =",
        p[dev].nTimeOff,
    )


####################################################
# Main function
####################################################

# Array of parameters for all the devices

p = []

# Create device instances in array
p.append(device("HFO1", MOTOR))
p.append(device("SGL", LAMP))

# Create server socket
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Created server socket")

"""
# Batch of commands for motor test (e.g. HFO1)
cmds = []
cmds.append("MAIN.HFO1.ctrl.bEnable=TRUE")
cmds.append("MAIN.HFO1.ctrl.nCommand=1")
cmds.append("MAIN.HFO1.ctrl.bExecute=TRUE")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.ctrl.lrPosition=5")
cmds.append("MAIN.HFO1.ctrl.nCommand=3")
cmds.append("MAIN.HFO1.ctrl.bExecute=TRUE")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")
cmds.append("MAIN.HFO1.stat.nAxisStatus")

n_cmd = 0
"""

running = 1

# Main loop

while running == 1:
    # Listen to ZMQ socket, parse message
    print("Listening to client...")
    try:
        message = socket.recv_string()
    except:
        print("closing socket...")
        socket.close()
        context.destroy()
        exit()
    """
    #message = input ("IN>")
    message = cmds[n_cmd]
    n_cmd = n_cmd + 1
    """
    print("IN>", message)
    if ("MAIN" in message) and (message.count(".") > 2):
        valid = True
    else:
        valid = False
    if valid:
        if "=" in message:
            # Received request is "write"
            x = re.split("=", message)
            read_val = x[1]
            y = re.split("\.", x[0])
            device = y[1]
            category = y[2]
            # Deal with the case of parameter arrays (two atoms)
            if len(y) == 5:
                # Replace dot and braces by underscores
                parameter = re.sub("\[|\]", "_", y[3]) + "_" + y[4]
                par_type = y[4][0]
            else:
                parameter = y[3]
                par_type = y[3][0]
            if par_type == "b":
                if read_val == "TRUE":
                    value = bool(1 > 0)
                else:
                    value = bool(1 < 0)
            if par_type == "n":
                value = int(read_val)
            if par_type == "l":
                value = float(read_val)
            # Search for device
            for d in range(2):
                if p[d].name == device:
                    break
            # Update parameter value of device
            print("parameter = ", parameter)
            if hasattr(p[d], parameter):
                setattr(p[d], parameter, value)

                if p[d]._devType == MOTOR:
                    update_motor_param(d)
                elif p[d]._devType == LAMP:
                    update_lamp_param(d)
                """ 
                elif p[d]._devType == SHUTTER:
                    update_shutter_params(p[d])
                elif p[d]._devType == SENSOR:
                    update_sensor_params(p[d])
                """
                # Send back acknowledgement to client
                print("Updated parameter", parameter, "of", device, "to", value)
            socket.send_string("ACK")
        else:
            # Received request is "read"
            x = re.split("\.", message)
            device = x[1]
            category = x[2]
            # Deal with the case of parameter arrays (two atoms)
            if len(x) == 5:
                # Replace dot and braces by underscores
                parameter = re.sub("\[|\]", "_", x[3]) + "_" + x[4]
                par_type = x[4][0]
            else:
                parameter = x[3]
                par_type = x[3][0]
            # Search for device
            for d in range(2):
                if p[d].name == device:
                    break
            if hasattr(p[d], parameter):
                value = getattr(p[d], parameter)
                if type(value) == int:
                    reply = "n" + str(value)
                elif type(value) == float:
                    reply = "r" + str(value)
                elif type(value) == bool:
                    if value == True:
                        reply = "bTRUE"
                    else:
                        reply = "bFALSE"
                elif type(value) == str:
                    reply = "s--UNKNOWN--"
            else:
                # Unknown parameter: send back garbage value according to type
                if par_type == "b":
                    reply = "bFALSE"
                elif par_type == "n":
                    reply = "n9999"
                elif par_type == "l":
                    reply == "r99.99"
                elif par_type == "s":
                    reply = "s--UNKNOWN--"
            if p[d]._devType == MOTOR:
                update_motor_fsm(d)
            elif p[d]._devType == LAMP:
                update_lamp_fsm(d)
            # Send reply to client
            print("Value of parameter", parameter, "of", device, "is:", value)
            print("OUT>", reply)
            socket.send_string(reply)
    else:
        # Garbage received => send anything to avoid blocking
        reply = "????"
        print("OUT>", reply)
        socket.send_string(reply)
    """
    print("----------------------")
    print("State:", p[0]._state)
    print("Enabled:", p[0].bEnabled)
    print("Initialised:", p[0].bInitialised)
    print("Axis Status:", p[0].nAxisStatus)
    print("Target position:", p[0].lrPosTarget)
    print("Actual position:", p[0].lrPosActual)
    print("----------------------")
    """
