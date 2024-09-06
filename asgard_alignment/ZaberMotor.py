"""
A module for controlling the Zaber motors: LAC10A-T4A (through a X-MCC), X-LSM and X-LHM

Need to come up with a way to be able to name an axis/optic and move the right controller
Ideas:
- XMCC class with usage like XMCC[<axis number>].move_absolute(1000), + a dictionary that maps 
    the name of the optic to both the axis number and controller
"""

import zaber_motion

import streamlit as st
from zaber_motion.ascii import Connection
import time
import datetime
import json

import zaber_motion.binary


tstamp = datetime.datetime.now().strftime("%d-%m-%YT%H.%M.%S") # for measuring drifts in phase mask when updating positions

class BifrostDichroic:
    def __init__(self, device) -> None:
        self.device = device
        self.axis = device.get_axis(1)
        self.dichroics = {
            "H": 133.07,  # 131.82,
            "J": 63.07,
            "out": 0.0,
        }

        assert self.device.name == "X-LSM150A-SE03"

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

        self.current_dichroic = "out"
        self.set_dichroic(self.current_dichroic)

    def set_dichroic(self, dichroic):
        """Move the optic to the desired position"""

        if dichroic not in self.dichroics:
            raise ValueError(f"Position {dichroic} not in {self.dichroics.keys()}")

        self.axis.move_absolute(
            self.dichroics[dichroic],
            unit=zaber_motion.Units.LENGTH_MILLIMETRES,
            wait_until_idle=False,
        )
        self.current_dichroic = dichroic

    def get_dichroic(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MILLIMETRES)
        for key, value in self.dichroics.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI(self):
        st.header("M100D motor")

        st.write(f"Current position: {self.get_dichroic()}")

        # 3 buttons for each position
        for key in self.dichroics.keys():
            if st.button(key):
                self.set_dichroic(key)


class SourceSelection:
    def __init__(self, device) -> None:
        self.device = device
        self.axis = device.get_axis(1)
        self.sources = {
            "SRL": 11.018 * 1e3,  # units converted mm->um
            "SGL": 38.018 * 1e3,
            "SBB": 65.018 * 1e3,
            "SLD": 92.018 * 1e3,
            "none": 0.0,
        }

        assert self.device.name == "X-LHM100A-SE03"

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

        self.current_source = "none"
        self.set_source(self.current_source)

    def set_source(self, source):
        """Move the optic to the desired position"""
        if source not in self.sources:
            raise ValueError(f"Position {source} not in {self.sources.keys()}")

        self.axis.move_absolute(
            self.sources[source],
            unit=zaber_motion.Units.LENGTH_MICROMETRES,
            wait_until_idle=True,
        )
        self.current_position = source

    def get_source(self):
        """Read the position from the device and check that it is consistent"""
        pos = self.axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        for key, value in self.sources.items():
            if abs(pos - value) < 0.1:
                return key
        return "unknown"

    def GUI(self):
        st.header("Source selection")

        st.write(f"Current position: {self.get_source()}")

        # 4 buttons for each position
        for key in self.sources.keys():
            if st.button(key):
                self.set_source(key)


class SolarsteinDelay:
    pass


class BaldrCommonLens:
    pass


class LAC10AT4A:
    def __init__(self, axis) -> None:
        self.axis = axis

        if not self.axis.is_homed:
            self.axis.home(wait_until_idle=True)

    def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        self.axis.move_absolute(new_pos, unit=units, wait_until_idle=True)

    def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
        self.axis.move_relative(new_pos, unit=units, wait_until_idle=True)

    def get_position(self, units=zaber_motion.Units.LENGTH_MICROMETRES):
        return self.axis.get_position(unit=units)


class BaldrPhaseMask:
    """
    Key here is that this has 2x LAC10A and can control both at once
    """

    def __init__(self, x_axis_motor, y_axis_motor, phase_positions_json) -> None:
        self.motors = {
            "x": x_axis_motor,
            "y": y_axis_motor,
        }

        self.phase_positions = self._load_phase_positions(phase_positions_json)
        
        #self._load_phasemask_parameters("phasemask_parameters_beam_3.json"):
        self.phasemask_parameters = {
                        "J1": {"depth":0.474 ,  "diameter":54},  
                        "J2": {"depth":0.474 ,  "diameter":44}, 
                        "J3": {"depth":0.474 ,  "diameter":36}, 
                        "J4": {"depth":0.474 ,  "diameter":32},
                        "J5": {"depth":0.474 ,  "diameter":65},
                        "H1": {"depth":0.654 ,  "diameter":68},  
                        "H2": {"depth":0.654 ,  "diameter":53}, 
                        "H3": {"depth":0.654 ,  "diameter":44}, 
                        "H4": {"depth":0.654 ,  "diameter":37},
                        "H5": {"depth":0.654 ,  "diameter":31}
                        }

    @staticmethod
    def _load_phase_positions(phase_positions_json):
        # all units in micrometers
        with open(phase_positions_json, "r", encoding="utf-8") as file:
            config = json.load(file)

        assert len(config) == 10, "There must be 10 phase mask positions"

        return config

    def _load_phasemask_parameters(phasemask_properties_json):
        # all units in micrometers
        with open(phase_positions_json, "r", encoding="utf-8") as file:
            config = json.load(file)

        assert len(config) == 10, "There must be 10 phase masks"

        return config
    

    def move_relative(self, new_pos, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        self.motors["x"].move_relative(new_pos[0], units)
        self.motors["y"].move_relative(new_pos[1], units)

    def move_absolute(self, new_pos, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        self.motors["x"].move_absolute(new_pos[0], units)
        self.motors["y"].move_absolute(new_pos[1], units)

    def get_position(self, units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        return [
            self.motors["x"].get_position(units),
            self.motors["y"].get_position(units),
        ]

    def move_to_mask(self, mask_name):
        self.move_absolute(self.phase_positions[mask_name])

    def update_mask_position(self, mask_name):
        self.phase_positions[mask_name] = self.get_position()

    def write_current_mask_positions( self , file_name=f'phase_positions_beam_3_{tstamp}.json'):
        with open(file_name, 'w') as f:
            json.dump(self.phase_positions, f)


    def update_all_mask_positions_relative_to_current(self, current_mask_name, reference_mask_position_file, write_file = False):
        # read in reference mask position file (any file where relative distances between masks is well calibrated - absolute values don't matter)
        # subtract off current_mask position in reference from all entries to generate offsets (so current_mask_name is origin in the offsets)
        # for each phase mask apply the relative offset from the current motor position to calculate new positions. update self.phase_positions.

        # read in positions from reference file
        reference_position = self._load_phase_positions(reference_mask_position_file)

        # set origin at the current phase mask in reference file 
        new_origin = reference_position[current_mask_name] 

        # get mapping (offsets) between phase masks from reference file relative to current_mask_name
        offsets = {}
        for mask_name, mask_position in reference_position.items():
            offsets[mask_name] = np.array( mask_position ) - np.array( new_origin ) 
            
        # apply offsets relative to the actual current motor position
        current_position = np.array( self.get_position() )
        for mask_name, offset in offsets.items() :  
            self.phase_positions[mask_name] = list( current_position + offset )

        if write_file:
            write_current_mask_positions( self , file_name=f'phase_positions_beam_3_{tstamp}.json')


if __name__ == "__main__":

    con = Connection.open_tcp("192.168.1.111")

    print("Found {} devices".format(len(con.detect_devices())))

    x_axis = con.get_device(1).get_axis(1)
    y_axis = con.get_device(1).get_axis(3)

    baldr = BaldrPhaseMask(
        LAC10AT4A(x_axis), LAC10AT4A(y_axis), "phase_positions_beam_3.json"
    )

    print(baldr.get_position())

    baldr.move_relative([0.1, 0.1])
    print(baldr.get_position())

    exit()
    connection = Connection.open_serial_port("COM3")
    connection.enable_alerts()

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))

    dichroics = []
    source_selection = None
    for dev in device_list:
        if dev.name == "X-LSM150A-SE03":
            dichroics.append(BifrostDichroic(dev))
        elif dev.name == "X-LHM100A-SE03":
            source_selection = SourceSelection(dev)
    print(f"Found {len(dichroics)} dichroics")
    if source_selection is not None:
        print("Found source selection")

    for dichroic in dichroics:
        dichroic.set_dichroic("J")

    while dichroics[0].get_dichroic() != "J":
        pass

    time.sleep(0.5)
    for dichroic in dichroics:
        print(dichroic.get_dichroic())

    for i in range(10):
        time.sleep(0.5)

        pos = dichroics[0].axis.get_position(unit=zaber_motion.Units.LENGTH_MICROMETRES)
        print(f"position: {pos:.3f}mm")

    source_selection.set_source("SRL")

    while source_selection.get_source() != "SRL":
        pass

    time.sleep(0.5)
    print(source_selection.get_source())

    connection.close()
