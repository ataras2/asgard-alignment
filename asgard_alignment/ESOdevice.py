"""
Scaffolds for devices to run with ESO command sets
"""

# abstract base class
import abc
from enum import Enum

import math
from datetime import datetime


class ESOdevice(abc.ABC):
    class DeviceType(Enum):
        MOTOR = 0
        SHUTTER = 1
        LAMP = 2
        SENSOR = 3
        IODEV = 4

    def __init__(self, name, dev_type) -> None:
        super().__init__()

        self.name = name
        self._dev_type = dev_type

        # status parameters (all devices)
        self.bLocal = False
        self.bInitialised = False

        self.nErrorCode = 0

        # internal parameters (used by the finite-state machines)
        self._state = 0
        self._t_last = 0
        self._t_on = 0
        self._t_off = 0

    @abc.abstractmethod
    def update_fsm(self):
        pass

    @abc.abstractmethod
    def update_param(self):
        pass


class Motor(ESOdevice):
    class Status(Enum):
        MOT_ERROR = 0
        MOT_STANDING = 1
        MOT_MOVING = 2
        MOT_INITIALISING = 3

    # Motor commands
    class Command(Enum):
        MOT_NONE = 0
        MOT_INITIALISE = 1
        MOT_SETPOS = 2
        MOT_MOVEABS = 3
        MOT_MOVEREL = 4
        MOT_MOVEVEL = 5
        MOT_NEWVEL = 6
        MOT_NEWPOS = 7

    # Motor states
    class State(Enum):
        IDLE = 0
        RESET_AXIS = 1
        MOVE = 2
        STOP = 3
        INIT = 4

    def __init__(self, name):
        dev_type = ESOdevice.DeviceType.MOTOR
        super().__init__(name, dev_type)

        self.nCommand = Motor.Command.MOT_NONE
        self.nLastCommand = Motor.Command.MOT_NONE

        self.bExecute = False

        # control parameters (motors)
        self.lrPosition = 0.0
        self.lrVelocity = 0.0
        self.bStop = False
        self.bDisable = False
        self.bEnable = False
        self.bResetError = False

        # status parameters (motors)
        self.lrScaleFactor = 1.0
        self.lrPosTarget = 10.0
        self.lrPosActual = 0.0
        self.lrBacklash = 0.0
        self.nAxisStatus = Motor.Status.MOT_STANDING
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

    @abc.abstractmethod
    def is_at_limit(self):
        pass

    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def move_abs(self, position):
        pass

    @abc.abstractmethod
    def is_reset_success(self):
        pass

    @abc.abstractmethod
    def is_stop_success(self):
        pass

    @abc.abstractmethod
    def is_reset_success(self):
        pass

    @abc.abstractmethod
    def is_init_success(self):
        pass

    @abc.abstractmethod
    def read_position(self):
        pass

    @abc.abstractmethod
    def is_motion_done(self):
        pass

    def update_param(self):
        if self.bEnable == True:
            self.bEnabled = True
            self.bEnable = False
        if self.bDisable == True:
            self.bEnabled = False
            self.bDisable = False
        #    if m.bEnabled == True and m.bInitialized == True:

        # ***********************************************
        # * ADD HERE CALL TO CHECK IF MOTOR IS AT LIMIT *
        # ***********************************************
        # If motor (device indexed by D) is at limit,
        #       enable and execute the following code:

        if self.is_at_limit():
            self.bEnabled = False
            self.nAxisStatus = Motor.Status.MOT_ERROR
            # TODO self.nErrorCode = MOT_ERROR_LIMIT ???
            self._state = Motor.State.IDLE

        if self.bStop == True:
            self.bStop = False
            # *********************************************
            # * ADD HERE CALL TO STOP MOTOR (PANIC STOP) *
            # *********************************************

            self._state = Motor.State.STOP

        if (
            self.bResetError == True
        ):  # unclear in which case this can actually be sent...
            # Reset axis
            self.bResetError = False
            self.bStop = False
            self.bInitialised = False
            self.nErrorCode = 0
            self._state = Motor.State.RESET_AXIS

        if self.bExecute == True:
            print("Executing command...")
            # Execute the last command received (as a parameter)
            self.nLastCommand = self.nCommand
            self.bExecute = False
            self.nCommand = Motor.Command.MOT_NONE
            if not self.bEnabled or (
                self.nLastCommand != Motor.Command.MOT_INITIALISE
                and not self.bInitialised
            ):
                print("Error: motor not initialised")
                # Error: motor not initialised
                self.nAxisStatus = Motor.Status.MOT_ERROR
                self.nErrorCode = 1000
                self._state = Motor.State.IDLE
            else:
                if self.nLastCommand == Motor.Command.MOT_INITIALISE:
                    # Initialise motor
                    self.bInitialised = False
                    self.nErrorCode = 0
                    self.nAxisStatus = Motor.Status.MOT_INITIALISING
                    print("Initialising")

                    # ***********************************************
                    # * ADD HERE CALL TO START MOTOR INITIALISATION *
                    # *
                    # * If the motors do not need initialisation by
                    # * the wag ICS, ignore this but keep the
                    # * code that modifies the self parameters
                    # ***********************************************
                    self.init()

                    self._state = Motor.State.INIT
                if self.nLastCommand == Motor.Command.MOT_MOVEABS:
                    # Move motor to absolute position
                    self.nErrorCode = 0
                    self.nAxisStatus = Motor.Status.MOT_MOVING
                    self.lrPosTarget = self.lrPosition

                    # *************************************************
                    # ADD HERE CALL TO START MOTOR MOTION TO ABS ENC *
                    # *************************************************
                    self.move_abs(self.lrPosTarget)

                    self._state = Motor.State.MOVE

                if self.nLastCommand == Motor.Command.MOT_MOVEREL:
                    # Move motor to relative position
                    self.nErrorCode = 0
                    self.nAxisStatus = Motor.Status.MOT_MOVING
                    # Convert relative move to absolute move
                    self.lrPosTarget = self.lrPosition + self.lrPosActual

                    # *************************************************
                    # ADD HERE CALL TO START MOTOR MOTION TO ABS ENC *
                    # *************************************************
                    self.move_abs(self.lrPosTarget)

                    self._state = Motor.State.MOVE
                    # the following line is only used for the built-in SW simulation
                    self._t_last = math.floor(datetime.timestamp(datetime.now()))

    def update_fsm(self):
        if self._state == Motor.State.IDLE:
            self.bResetError = False
            self.bStop = False

        elif self._state == Motor.State.RESET_AXIS:
            if self.is_reset_success():
                self.nErrorCode = 0
                self.nAxisStatus = Motor.Status.MOT_STANDING
                self._state = Motor.State.IDLE
            else:
                self.nAxisStatus = Motor.Status.MOT_ERROR
                self._state = Motor.State.IDLE

        elif self._state == Motor.State.STOP:
            if self.is_stop_success():
                self.nErrorCode = 0
                self.nAxisStatus = Motor.Status.MOT_STANDING
                self._state = Motor.State.IDLE
            else:
                self.nAxisStatus = Motor.Status.MOT_ERROR
                self._state = Motor.State.IDLE
                # TODO self.nErrorCode = MOT_ERROR_STOP ???

        elif self._state == Motor.State.INIT:
            if self.is_init_success():
                self.nErrorCode = 0
                self.nAxisStatus = Motor.Status.MOT_STANDING
                self.bInitialised = True
                self._state = Motor.State.IDLE
            else:
                self.nAxisStatus = Motor.Status.MOT_ERROR
                self._state = Motor.State.IDLE
                # TODO self.nErrorCode = MOT_ERROR_INIT ???
                self.bInitialised = False

        elif self._state == Motor.State.MOVE:
            self.lrPosActual = self.read_position()

            # *********************************************
            # * ADD HERE CALL TO CHECK STATUS OF MOTION   *
            # *********************************************
            #     If motion failed (and motor stopped), execute the following:
            #     self.nErrorCode = <the error code>
            #     self.nAxisStatus = MOT_ERROR
            #     self._state = IDLE
            #
            #     If motion succeeded, execute the following:

            if self.is_motion_done() == True:
                self.nErrorCode = 0
                self.nAxisStatus = Motor.Status.MOT_STANDING
                self.bInitialised = True
                self._state = Motor.State.IDLE
            else:
                self.nAxisStatus = Motor.Status.MOT_ERROR
                self._state = Motor.State.IDLE
                # TODO self.nErrorCode = MOT_ERROR_MOVE ???


class Lamp(ESOdevice):
    # Codes for lamp status
    class Status(Enum):
        LMP_ISOFF = 0
        LMP_WARMING_UP = 1
        LMP_ISON = 2
        LMP_COOLING_DOWN = 3
        LMP_ERROR = 4

    # Lamp commands
    class Command(Enum):
        LMP_NONE = 0
        LMP_INITIALISE = 1
        LMP_OFF = 2
        LMP_ON = 3

    # Lamp error codes
    class ErrorCode(Enum):
        LMP_OK = 0
        LMP_FAULT_SIG = 1
        LMP_NOT_IN_OP_STATE = 2
        LMP_MAXON_TIMEOUT = 3
        LMP_NOT_COOLED = 4
        LMP_LOCAL_MODE = 5
        LMP_WHILE_ON_WENT_OFF = 6
        LMP_NO_FEEDBACK_SIG = 7
        LMP_NOT_INITIALISED = 10

    def __init__(self, name):
        super().__init__(name, ESOdevice.DeviceType.LAMP)

        self.bExecute = False

        self.nCommand = Lamp.Status.LMP_NONE

        self.lrIntensity = 100.0
        self.nTimeOff = 0  # time in seconds since lamp was turned off
        self.nTimeOn = 0  # time in seconds since lamp was turned on

        self.nCooldown = (
            5  # time in seconds to wait before turning lamp on again, can be 0
        )
        self.nWarmup = 5  # time in seconds to wait before lamp is fully on
        self.nMaxOn = 4000

    @abc.abstractmethod
    def turn_on(self):
        pass

    @abc.abstractmethod
    def turn_off(self):
        pass

    @abc.abstractmethod
    def init(self):
        pass

    def update_param(self):
        if self.bExecute == True:
            self.nLastCommand = self.nCommand
            self.bExecute = False
            self.nCommand = Lamp.Status.LMP_NONE
            if self.nLastCommand == Lamp.Status.LMP_INITIALISE:
                if self.nTimeOn > 0:
                    # ***************************************************
                    # ADD HERE CALL TO GET LAMP INITIALISED (if needed)
                    # AND OFF
                    # ***************************************************
                    self._t_off = math.floor(datetime.timestamp(datetime.now()))
                    self.nStatus = Lamp.Status.LMP_COOLING_DOWN
                else:
                    self.nStatus = Lamp.Status.LMP_ISOFF
                    self.nErrorCode = Lamp.Status.LMP_OK
                    self.bInitialised = True
                    self._t_off = math.floor(datetime.timestamp(datetime.now()))
            else:
                if self.bInitialised == True:
                    if self.nLastCommand == Lamp.Status.LMP_OFF:
                        # ************************************
                        # ADD HERE CALL TO TURN LAMP OFF
                        # ************************************
                        print("Turning lamp off...")
                        self._t_off = math.floor(datetime.timestamp(datetime.now()))
                        self.nTimeOn = 0
                        self.nTimeOff = 0
                        self.nErrorCode = Lamp.Status.LMP_OK
                        self.nStatus = Lamp.Status.LMP_COOLING_DOWN
                    if self.nLastCommand == Lamp.Status.LMP_ON:
                        t_now = math.floor(datetime.timestamp(datetime.now()))
                        # Check lamp is cold before turning it on again
                        # (to prevent damaging for some lamps)
                        if (
                            self.nStatus == Lamp.Status.LMP_COOLING_DOWN
                            or (t_now - self._t_off) < self.nCooldown
                        ):
                            self.nErrorCode = Lamp.Status.LMP_NOT_COOLED
                            self.nStatus = Lamp.Status.LMP_ERROR
                        else:
                            # ********************************
                            # ADD HERE CALL TO TURN LAMP ON
                            # ********************************
                            print("Turning lamp on...")
                            self._t_on = math.floor(datetime.timestamp(datetime.now()))
                            self.nTimeOn = 0
                            self.nTimeOff = 0
                            self.nErrorCode = Lamp.Status.LMP_OK
                            self.nStatus = Lamp.Status.LMP_WARMING_UP

    def update_fsm(self):
        t_now = math.floor(datetime.timestamp(datetime.now()))
        if self.nStatus == Lamp.Status.LMP_WARMING_UP:
            self.nTimeOn = t_now - self._t_on
            if self.nTimeOn > self.nWarmup:
                self.nStatus = Lamp.Status.LMP_ISON
        elif self.nStatus == Lamp.Status.LMP_ISON:
            self.nTimeOn = t_now - self._t_on
            if self.nTimeOn > self.nMaxOn:
                # *******************************************
                # ADD HERE CALL TO TURN LAMP OFF FOR SAFETY
                # (lamp has been on for too long)
                # *******************************************
                self._t_off = math.floor(datetime.timestamp(datetime.now()))
                self.nTimeOff = 0
                self.nTimeOn = 0
                self.nStatus = Lamp.Status.LMP_COOLING_DOWN
        elif self.nStatus == Lamp.Status.LMP_COOLING_DOWN:
            self.nTimeOff = t_now - self._t_off
            if self.nTimeOff > self.nCooldown:
                self.nStatus = Lamp.Status.LMP_ISOFF
        if self.nStatus == Lamp.Status.LMP_ISOFF:
            self.nTimeOff = t_now - self._t_off
        print(
            "Lamp state is",
            self.nStatus,
            " ; tOn =",
            self.nTimeOn,
            " ; tOff =",
            self.nTimeOff,
        )
