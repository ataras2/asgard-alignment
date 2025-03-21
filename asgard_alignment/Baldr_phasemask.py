import json
import zaber_motion  # .binary
import numpy as np
import datetime
import os

# import argparse
# import zmq


class BaldrPhaseMask:
    """
    Key here is that this has 2x LAC10A and can control both at once
    """

    def __init__(self, beam, x_axis_motor, y_axis_motor, phase_positions_json) -> None:
        self.motors = {
            "x": x_axis_motor,
            "y": y_axis_motor,
        }

        cnt_pth = os.path.dirname(os.path.abspath(__file__))
        save_path = cnt_pth + os.path.dirname("/../config_files/phasemask_positions/")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.savepath = save_path  # where we save updated position json files

        self.beam = beam
        self.phase_positions = self._load_phase_positions(phase_positions_json)

        # self._load_phasemask_parameters("phasemask_parameters_beam_3.json"):
        self.phasemask_parameters = {
            "J1": {"depth": 0.474, "diameter": 54},
            "J2": {"depth": 0.474, "diameter": 44},
            "J3": {"depth": 0.474, "diameter": 36},
            "J4": {"depth": 0.474, "diameter": 32},
            "J5": {"depth": 0.474, "diameter": 65},
            "H1": {"depth": 0.654, "diameter": 68},
            "H2": {"depth": 0.654, "diameter": 53},
            "H3": {"depth": 0.654, "diameter": 44},
            "H4": {"depth": 0.654, "diameter": 37},
            "H5": {"depth": 0.654, "diameter": 31},
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
        with open(phasemask_properties_json, "r", encoding="utf-8") as file:
            config = json.load(file)

        assert len(config) == 10, "There must be 10 phase masks"

        return config

    def move_relative(self, new_pos):
        self.motors["x"].move_relative(
            new_pos[0], units=zaber_motion.units.Units.LENGTH_MICROMETRES
        )  # moverel_msg(float(new_pos[0])), #
        self.motors["y"].move_relative(
            new_pos[1], units=zaber_motion.units.Units.LENGTH_MICROMETRES
        )  # moverel_msg(float(new_pos[1])) #

    def move_absolute(self, new_pos):
        self.motors["x"].move_absolute(
            new_pos[0], units=zaber_motion.units.Units.LENGTH_MICROMETRES
        )  # moveabs_msg(float(new_pos[0]))#
        self.motors["y"].move_absolute(
            new_pos[1], units=zaber_motion.units.Units.LENGTH_MICROMETRES
        )  # moveabs_msg(float(new_pos[1]))

    def read_position(self):  # , units=zaber_motion.units.Units.LENGTH_MICROMETRES):
        return [
            self.motors["x"].read_position(
                units=zaber_motion.units.Units.LENGTH_MICROMETRES
            ),  # read_msg(), #
            self.motors["y"].read_position(
                units=zaber_motion.units.Units.LENGTH_MICROMETRES
            ),  # read_msg(),
        ]

    def move_to_mask(self, mask_name):
        self.move_absolute(self.phase_positions[mask_name])

    def update_mask_position(self, mask_name):
        self.phase_positions[mask_name] = self.read_position()

    def write_current_mask_positions(self):
        tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        file_name = (
            self.savepath
            + f"/beam{self.beam}/phase_positions_beam{self.beam}_{tstamp}.json"
        )

        with open(file_name, "w") as f:
            json.dump(self.phase_positions, f, indent=4)

    def offset_all_mask_positions(self, rel_offset_x, rel_offset_y):
        "offset ALL phasemask positions by rel_offset_x, rel_offset_y"
        for mask in self.phase_positions:
            self.phase_positions[mask][0] += rel_offset_x
            self.phase_positions[mask][1] += rel_offset_y
        


    def update_all_mask_positions_relative_to_current(
        self, current_mask_name, reference_mask_position_file, write_file=False
    ):
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
            offsets[mask_name] = np.array(mask_position) - np.array(new_origin)

        # apply offsets relative to the actual current motor position
        current_position = np.array(self.read_position())
        for mask_name, offset in offsets.items():
            self.phase_positions[mask_name] = list(current_position + offset)

        if write_file:
            tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.write_current_mask_positions(
                file_name=self.savepath
                + f"phase_positions_beam_{self.beam}_{tstamp}.json"
            )


# ======================================
# below is testing class with ZMQ server
## To DO: pass socket as argument to class init??

## ensure MultiDeviceServer is running first!! If not run the following command in terminal from asgard_alignment directory:
## asgard_alignment/MultiDeviceServer.py -c motor_info_full_system_with_DMs.json

# parser = argparse.ArgumentParser(description="ZeroMQ Client")
# parser.add_argument("--host", type=str, default="localhost", help="Server host")
# parser.add_argument("--port", type=int, default=5555, help="Server port")
# parser.add_argument("--timeout", type=int, default=5000, help="Response timeout in milliseconds")

# parser.parse_args()

# args = parser.parse_args()

# context = zmq.Context()

# context.socket(zmq.REQ)

# socket = context.socket(zmq.REQ)

# socket.setsockopt( zmq.RCVTIMEO, args.timeout )

# server_address = (f"tcp://{args.host}:{args.port}")

# socket.connect(server_address)

# state_dict = {"message_history": [], "socket": socket}


# def send_and_get_response(message):
#     # st.write(f"Sending message to server: {message}")
#     state_dict["message_history"].append(
#         f":blue[Sending message to server: ] {message}\n"
#     )
#     state_dict["socket"].send_string(message)
#     response = state_dict["socket"].recv_string()
#     if "NACK" in response or "not connected" in response:
#         colour = "red"
#     else:
#         colour = "green"
#     # st.markdown(f":{colour}[Received response from server: ] {response}")
#     state_dict["message_history"].append(
#         f":{colour}[Received response from server: ] {response}\n"
#     )

#     return response.strip()


# class BaldrPhaseMask:
#     """
#     ensure MultiDeviceServer is running first!! If not run the following command in terminal from asgard_alignment directory:
#         > asgard_alignment/MultiDeviceServer.py -c motor_info_full_system_with_DMs.json

#     uses zmq to communicate with the server to control the device
#     uses convention that the Zaber motors in MultiDeviceServer are named BMX1, BMY1, BMX2, BMY2, etc.
#     where X, Y are X and Y coordinates and 1, 2, 3, 4 are the beam numbers.
#     """

#     def __init__(self, beam, phase_positions_json ) -> None:  #, x_axis_motor, y_axis_motor, phase_positions_json) -> None:

#         self.beam = beam

#         self.phase_positions = self._load_phase_positions(phase_positions_json)

#         #self._load_phasemask_parameters("phasemask_parameters_beam_3.json"):
#         self.phasemask_parameters = {
#                         "J1": {"depth":0.474 ,  "diameter":54},
#                         "J2": {"depth":0.474 ,  "diameter":44},
#                         "J3": {"depth":0.474 ,  "diameter":36},
#                         "J4": {"depth":0.474 ,  "diameter":32},
#                         "J5": {"depth":0.474 ,  "diameter":65},
#                         "H1": {"depth":0.654 ,  "diameter":68},
#                         "H2": {"depth":0.654 ,  "diameter":53},
#                         "H3": {"depth":0.654 ,  "diameter":44},
#                         "H4": {"depth":0.654 ,  "diameter":37},
#                         "H5": {"depth":0.654 ,  "diameter":31}
#                         }

#     @staticmethod
#     def _load_phase_positions(phase_positions_json):
#         # all units in micrometers
#         with open(phase_positions_json, "r", encoding="utf-8") as file:
#             config = json.load(file)

#         assert len(config) == 10, "There must be 10 phase mask positions"

#         return config

#     def _load_phasemask_parameters(phasemask_properties_json):
#         # all units in micrometers
#         with open(phase_positions_json, "r", encoding="utf-8") as file:
#             config = json.load(file)

#         assert len(config) == 10, "There must be 10 phase masks"

#         return config


#     def move_relative(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
#         # new_pos is a list of 2 floats [x,y]
#         # NOTE: multidevice server custom commands require (12 Nov 24) format "!moveabs {} {:f}". ie. number must be float
#         # X-axis
#         message = f"!moverel BMX{self.beam} {float(new_pos[0])}"
#         res = send_and_get_response(message)
#         if "NACK" in res:
#             print(f"Error for {message}")
#             #break

#         # Y-axis
#         message = f"!moverel BMY{self.beam} {float(new_pos[1])}"
#         res = send_and_get_response(message)
#         if "NACK" in res:
#             print(f"Error for {message}")
#             #break


#     def move_absolute(self, new_pos, units=zaber_motion.Units.LENGTH_MICROMETRES):
#         # new_pos is a list of 2 floats [x,y]
#         # NOTE: multidevice server custom commands require (12 Nov 24) format "!moveabs {} {:f}". ie. pos number must be float
#         # X-axis
#         message = f"!moveabs BMX{self.beam} {float(new_pos[0])}"
#         res = send_and_get_response(message)
#         if "NACK" in res:
#             print(f"Error for {message}")
#             #break

#         # Y-axis
#         message = f"!moveabs BMY{self.beam} {float(new_pos[1])}"
#         res = send_and_get_response(message)
#         if "NACK" in res:
#             print(f"Error for {message}")
#             #break


#     def get_position(self, units=zaber_motion.Units.LENGTH_MICROMETRES):

#         res_X = send_and_get_response(f"!read BMX{self.beam}")
#         res_Y = send_and_get_response(f"!read BMY{self.beam}")
#         pos = [res_X, res_Y]

#         #return [
#         #    self.motors["x"].get_position(units),
#         #    self.motors["y"].get_position(units),
#         #]
#         return pos

#     def move_to_mask(self, mask_name):
#         self.move_absolute(self.phase_positions[mask_name])

#     def update_mask_position(self, mask_name):
#         self.phase_positions[mask_name] = self.get_position()

#     def write_current_mask_positions( self , file_name=None):
#         tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#         if file_name is None:
#             file_name = f'phase_positions_beam_{self.beam}_{tstamp}.json'
#         with open(file_name, 'w') as f:
#             json.dump(self.phase_positions, f)


#     def update_all_mask_positions_relative_to_current(self, current_mask_name, reference_mask_position_file, write_file = False):
#         # read in reference mask position file (any file where relative distances between masks is well calibrated - absolute values don't matter)
#         # subtract off current_mask position in reference from all entries to generate offsets (so current_mask_name is origin in the offsets)
#         # for each phase mask apply the relative offset from the current motor position to calculate new positions. update self.phase_positions.

#         # read in positions from reference file
#         reference_position = self._load_phase_positions(reference_mask_position_file)

#         # set origin at the current phase mask in reference file
#         new_origin = reference_position[current_mask_name]

#         # get mapping (offsets) between phase masks from reference file relative to current_mask_name
#         offsets = {}
#         for mask_name, mask_position in reference_position.items():
#             offsets[mask_name] = np.array( mask_position ) - np.array( new_origin )

#         # apply offsets relative to the actual current motor position
#         current_position = np.array( self.get_position() )
#         for mask_name, offset in offsets.items() :
#             self.phase_positions[mask_name] = list( current_position + offset )

#         if write_file:
#             write_current_mask_positions( self , file_name=f'phase_positions_beam_{self.beam}_{tstamp}.json')


# """
# message = "!read BMY1"
# send_and_get_response(message)

# socket.send_string(f"!moveabs {axis} {positions[0]}")
# res = socket.recv_string()
# print(f"Response: {res}")
# """
