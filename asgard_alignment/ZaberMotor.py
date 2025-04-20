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

    def __init__(self, name, semaphore_id, axis, named_positions=None) -> None:
        super().__init__(name, semaphore_id, named_positions)
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

        if self.axis.warnings.get_flags() == set():
            return "No error"
        return f"{self.axis.warnings.get_flags()}"

    def ping(self):
        try:
            self.axis.is_busy()
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

    @staticmethod
    def internal_to_ESO(value):
        """
        Device moves in um, ESO moves in um (discrete)
        """
        return int(value)

    @staticmethod
    def ESO_to_internal(value):
        """
        Device moves in um, ESO moves in um (discrete)
        """
        return float(value)

    def ESO_read_position(self):
        return self.internal_to_ESO(
            self.axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        )

    def setup(self, motion_type, value):
        if motion_type == "NAME":
            try:
                self.move_absolute(self.named_positions[value])
            except KeyError:
                print(f"{self.name} does not have a named position {value}")

            return

        value = self.ESO_to_internal(value)
        if motion_type == "ENC":
            self.move_absolute(value)
        elif motion_type == "ENCREL":
            self.move_relative(value)

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

    def __init__(self, name, semaphore_id, device, named_positions=None):
        super().__init__(name, semaphore_id, named_positions)
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
            self.axis.is_busy()
            return True
        except Exception:
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
        if self.device.warnings.get_flags() == set():
            return "No error"
        return f"{self.device.warnings.get_flags()}"

    def read_position(self, units=zaber_motion.Units.LENGTH_MILLIMETRES):
        return self.axis.get_position(unit=units)

    def ESO_read_position(self):
        return self.internal_to_ESO(
            self.axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        )

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

    @staticmethod
    def internal_to_ESO(value):
        """
        Device moves in mm, ESO moves in um (discrete)
        """
        return int(value * 1_000)

    @staticmethod
    def ESO_to_internal(value):
        """
        Device moves in um, ESO moves in mm (discrete)
        """
        return float(value) / 1_000

    def setup(self, motion_type, value):
        if motion_type == "NAME":
            try:
                self.move_absolute(self.named_positions[value])
            except KeyError:
                print(f"{self.name} does not have a named position {value}")

            return

        value = self.ESO_to_internal(value)
        if motion_type == "ENC":
            self.move_absolute(value)
        elif motion_type == "ENCREL":
            self.move_relative(value)

    def disable(self):
        pass

    def enable(self):
        pass

    def online(self):
        pass

    def standby(self):
        pass
