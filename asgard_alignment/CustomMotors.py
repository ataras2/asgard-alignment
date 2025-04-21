import asgard_alignment.ESOdevice
import numpy as np

import asgard_alignment.controllino


class PK2FVF1(asgard_alignment.ESOdevice.Motor):
    def __init__(self, name, semaphore_id, controllino_controller) -> None:
        super().__init__(name, semaphore_id, {})

        self._controller = controllino_controller


class MFF101(asgard_alignment.ESOdevice.Motor):
    def __init__(
        self,
        name: str,
        semaphore_id: int,
        controllino_controller: asgard_alignment.controllino.Controllino,
        named_pos: dict,
    ) -> None:
        super().__init__(
            name,
            semaphore_id,
            named_positions=named_pos,
        )

        self._controller = controllino_controller

    def move_abs(self, position):
        if np.isclose(position, 1.0):
            self._controller.turn_on(self.name)
        elif np.isclose(position, 0.0):
            self._controller.turn_off(self.name)
        else:
            raise ValueError(f"Invalid position for bistable motor {self.name}")

    def read_position(self):
        return str(float(self._controller.get_status(self.name)))

    def move_relative(self, position: float):
        pass

    def ping(self):
        pass

    def is_moving(self):
        return False

    def read_state(self):
        return str(self._controller.get_status(self.name))

    def setup(self, motion_type, value):

        if motion_type == "ENC":
            self.move_abs(value)
        elif motion_type == "ENCREL":
            print(f"ERROR: ENCREL not implemented for {self.name}")
        elif motion_type == "NAME":
            self.move_abs(self.named_positions[value])

    def ESO_read_position(self):
        return int(float(self.read_position()))

    def disable(self):
        pass

    def enable(self):
        pass

    def stop(self):
        pass

    def online(self):
        pass

    def standby(self):
        pass


class MirrorFlipper(asgard_alignment.ESOdevice.Motor):
    def __init__(
        self,
        name,
        semaphore_id,
        controllino_controller,
        modulation_value,
        delay_time,
    ) -> None:
        named_pos = {"OUT": 0, "IN": 1}
        super().__init__(
            name,
            semaphore_id,
            named_positions=named_pos,
        )

        self._controller = controllino_controller
        self._state = ""

        self._modulation_value = modulation_value
        self._delay_time = delay_time

    def _flip_up(self):
        self._controller.flip_up(self.name, self._modulation_value, self._delay_time)
        self._state = "IN"

    def _flip_down(self):
        self._controller.flip_down(self.name, self._modulation_value, self._delay_time)
        self._state = "OUT"

    def move_abs(self, position):
        print(f"Moving {self.name} to {position}")
        if isinstance(position, str):
            position = self.named_positions[position]
        position = int(position)

        if position == 0:
            self._flip_down()
        if position == 1:
            self._flip_up()

    def move_relative(self, position: float):
        print(f"Move_rel not implemented for {self.name}")

    def read_position(self):
        return self._state

    def is_moving(self):
        return False

    def ESO_read_position(self):
        return self.read_position()

    def read_state(self):
        return f"READY ({self._state})"

    def setup(self, motion_type, value):
        if motion_type == "ENC":
            self.move_abs(value)
        elif motion_type == "ENCREL":
            print(f"ERROR: ENCREL not implemented for {self.name}")
        elif motion_type == "NAME":
            self.move_abs(self.named_positions[value])
        else:
            print(f"ERROR: Unknown motion type {motion_type} for {self.name}")

    def disable(self):
        pass

    def enable(self):
        pass

    def stop(self):
        pass

    def online(self):
        pass

    def standby(self):
        pass

    def ping(self):
        return True
