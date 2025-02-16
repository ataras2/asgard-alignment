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
        if isinstance(position, str):
            position = self._named_positions[position]

        if np.isclose(position, 1.0):
            self._controller.turn_on(self.name)
        elif np.isclose(position, 0.0):
            self._controller.turn_off(self.name)
        else:
            raise ValueError(f"Invalid position for bistable motor {self.name}")

    def read_position(self):
        self._controller.get_status(self.name)

    def move_relative(self, position: float):
        pass

    def ping(self):
        pass

    def read_state(self):
        pass

    def setup(self, value):
        self.move_abs(value)

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
        named_pos = {"down": 0, "up": 1}
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
        self._state = "up"

    def _flip_down(self):
        self._controller.flip_down(self.name, self._modulation_value, self._delay_time)
        self._state = "down"

    def move_abs(self, position):
        print(f"Moving {self.name} to {position}")
        if isinstance(position, str):
            position = self._named_positions[position]
        position = int(position)

        if position == 0:
            self._flip_down()
        if position == 1:
            self._flip_up()

    def move_relative(self, position: float):
        pass

    def read_position(self):
        return self._state

    def read_state(self):
        return f"READY ({self._state})"

    def setup(self, value):
        pass

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
