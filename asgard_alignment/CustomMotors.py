import asgard_alignment.ESOdevice


class PK2FVF1(asgard_alignment.ESOdevice.Motor):
    def __init__(self, name, semaphore_id, controllino_controller) -> None:
        super().__init__(name, semaphore_id, {})

        self._controller = controllino_controller


class MFF101(asgard_alignment.ESOdevice.Motor):
    def __init__(self, name, semaphore_id, controllino_controller) -> None:
        named_pos = {"30mm": 0, "15mm": 1}
        super().__init__(
            name,
            semaphore_id,
            named_positions=named_pos,
        )

        self._controller = controllino_controller

    def move_abs(self, position: float):
        pass

    def move_relative(self, position: float):
        pass

    def read_state(self):
        pass

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
