"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps the name of the optic to both the axis number and controller
"""

import zaber_motion

import streamlit as st
from zaber_motion.ascii import Connection
import time
import json
import numpy as np
import ast

import zaber_motion.binary


import asgard_alignment.ESOdevice as ESOdevice


import asgard_alignment.ESOdevice as ESOdevice


class ZaberLinearActuator(ESOdevice.Motor):
    UPPER_LIMIT = 10_000  # um
    LOWER_LIMIT = 0  # um

    IS_BLOCKING = False

    def __init__(self, name, semaphore_id, axis) -> None:
        super().__init__(name, semaphore_id)
        self.axis = axis

        if not self.axis.is_homed():
            self.axis.home(wait_until_idle=ZaberLinearActuator.IS_BLOCKING)

    def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        new_pos: float
            The position to move to

        units: zaber_motion.Units
            The units of the position, default is micrometres

        Returns:
        --------
        None
        """
        if self.LOWER_LIMIT <= new_pos <= self.UPPER_LIMIT:
            self.axis.move_absolute(
                new_pos,
                unit=units,
                wait_until_idle=ZaberLinearActuator.IS_BLOCKING,
            )

    def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        """
        Move the motor to the relative position

        Parameters:
        -----------
        new_pos: float
            The position to move to, relative to the current position

        units: zaber_motion.Units
            The units of the position, default is micrometres

        Returns:
        --------
        None
        """
        self.axis.move_relative(
            new_pos,
            unit=units,
            wait_until_idle=ZaberLinearActuator.IS_BLOCKING,
        )

    def read_position(self, units=zaber_motion.Units.LENGTH_MICROMETRES):
        return self.axis.get_position(unit=units)

    def is_at_limit(self):
        """
        Check if the motor is at the limit

        Returns:
        --------
        bool
            True if the motor is at the limit, False otherwise
        """
        # us np.isclose to avoid floating point errors
        return np.isclose(self.read_position(), self.UPPER_LIMIT) or np.isclose(
            self.read_position(), self.LOWER_LIMIT
        )

    def read_state(self):
        """
        Read the state of the motor

        Returns:
        --------
        str
            The state of the motor
        """
        # return self.axis.get_state()
        return f"Warnings/errors: {self.axis.warnings.get_flags()}"

    def ping(self):
        try:
            self.axis.get_device_id()
            return True
        except Exception as e:
            return False

    def stop_now(self):
        """
        Stop the motor immediately

        Returns:
        --------
        None
        """
        self.axis.stop(wait_until_idle=ZaberLinearActuator.IS_BLOCKING)

    def init(self):
        """
        Don't do anything, the motor is already initialised by the constructor
        Might need to home after power cycle
        """
        if not self.axis.is_homed():
            self.axis.home(wait_until_idle=ZaberLinearActuator.IS_BLOCKING)

    def move_abs(self, position):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        position: float
            The position to move to

        Returns:
        --------
        None
        """
        self.move_absolute(position)

    def is_reset_success(self):
        """
        Check if the reset was successful

        Returns:
        --------
        bool
            True if the reset was successful, False otherwise
        """

        return True  # TODO: Implement?

    def is_stop_success(self):
        """
        Check if the stop was successful

        Returns:
        --------
        bool
            True if the stop was successful, False otherwise
        """

        return not self.axis.is_busy()

    def is_init_success(self):
        """
        Check if the initialisation was successful

        Returns:
        --------
        bool
            True if the initialisation was successful, False otherwise
        """
        return not self.axis.is_busy()

    def is_motion_done(self):
        """
        Check if the motion is done

        Returns:
        --------
        bool
            True if the motion is done, False otherwise
        """
        return not self.axis.is_busy()

    def stop(self):
        self.axis.stop()

    def setup(self, value):
        self.axis.move_absolute(value)

    def disable(self):
        pass

    def enable(self):
        pass

    def online(self):
        pass

    def standby(self):
        pass


class ZaberLinearStage(ESOdevice.Motor):
    """
    A linear stage, e.g. the X-LHM100A-SE03, or the X-LSM150A-SE03
    Default units are milimetres
    """

    IS_BLOCKING = False

    def __init__(self, name, semaphore_id, device):
        super().__init__(name, semaphore_id)
        self.device = device
        self.axis = device.get_axis(1)

        # get the device type and the bounds from it
        if self.device.name == "X-LHM100A-SE03":
            self.UPPER_LIMIT = 100
            self.LOWER_LIMIT = 0
        elif self.device.name == "X-LSM150A-SE03":
            self.UPPER_LIMIT = 150
            self.LOWER_LIMIT = 0
        else:
            raise ValueError(f"Unknown device {self.device.name}")

        self.init()

    def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MILLIMETRES):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        new_pos: float
            The position to move to

        units: zaber_motion.Units
            The units of the position, default is milimetres

        Returns:
        --------
        None
        """
        self.axis.move_absolute(
            new_pos,
            unit=units,
            wait_until_idle=ZaberLinearStage.IS_BLOCKING,
        )

    def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MILLIMETRES):
        """
        Move the motor to the relative position

        Parameters:
        -----------
        new_pos: float
            The position to move to, relative to the current position

        units: zaber_motion.Units
            The units of the position, default is milimetres

        Returns:
        --------
        None
        """
        self.axis.move_relative(
            new_pos,
            unit=units,
            wait_until_idle=ZaberLinearStage.IS_BLOCKING,
        )

    def ping(self):
        try:
            self.axis.get_device_id()
            return True
        except Exception as e:
            return False

    def read_state(self):
        """
        Read the state of the motor

        Returns:
        --------
        str
            The state of the motor
        """
        # return ast.literal_eval(self.device.get_state())
        return f"Warnings/errors: {self.device.warnings.get_flags()}"

    def read_position(self, units=zaber_motion.Units.LENGTH_MILLIMETRES):
        return self.axis.get_position(unit=units)

    def is_at_limit(self):
        """
        Check if the motor is at the limit

        Returns:
        --------
        bool
            True if the motor is at the limit, False otherwise
        """
        # us np.isclose to avoid floating point errors
        return np.isclose(self.read_position(), self.UPPER_LIMIT) or np.isclose(
            self.read_position(), self.LOWER_LIMIT
        )

    def stop_now(self):
        """
        Stop the motor immediately

        Returns:
        --------
        None
        """
        self.axis.stop(wait_until_idle=ZaberLinearStage.IS_BLOCKING)

    def init(self):
        """
        Don't do anything, the motor is already initialised by the constructor
        Might need to home after power cycle
        """
        if not self.axis.is_homed():
            self.axis.home(wait_until_idle=ZaberLinearStage.IS_BLOCKING)

    def move_abs(self, position):
        """
        Move the motor to the absolute position

        Parameters:
        -----------
        position: float
            The position to move to

        Returns:
        --------
        None
        """
        self.move_absolute(position)

    def is_reset_success(self):
        """
        Check if the reset was successful

        Returns:
        --------
        bool
            True if the reset was successful, False otherwise
        """

        return True

    def is_stop_success(self):
        """
        Check if the stop was successful

        Returns:
        --------
        bool
            True if the stop was successful, False otherwise
        """

        return not self.axis.is_busy()

    def is_init_success(self):
        """
        Check if the initialisation was successful

        Returns:
        --------
        bool
            True if the initialisation was successful, False otherwise
        """
        return not self.axis.is_busy()

    def is_motion_done(self):
        """
        Check if the motion is done

        Returns:
        --------
        bool
            True if the motion is done, False otherwise
        """
        return not self.axis.is_busy()

    def stop(self):
        pass

    def setup(self, value):
        pass

    def disable(self):
        pass

    def enable(self):
        pass

    def online(self):
        pass

    def standby(self):
        pass
