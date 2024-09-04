import pyvisa
import time

# print(pyvisa.ResourceManager().list_resources())

import asgard_alignment.NewportMotor as nm

motor = nm.LS16P('ASRL/dev/ttyACM0::INSTR', pyvisa.ResourceManager())
print(motor.query_str("SA?"))

motor = nm.LS16P('ASRL/dev/ttyACM1::INSTR', pyvisa.ResourceManager())
print(motor.query_str("SA?"))

motor = nm.LS16P('ASRL/dev/ttyACM2::INSTR', pyvisa.ResourceManager())
print(motor.query_str("SA?"))

motor = nm.LS16P('ASRL/dev/ttyACM3::INSTR', pyvisa.ResourceManager())
print(motor.query_str("SA?"))

# to change a controller address:
# print(motor.query_str("SA?"))

# motor.write_str("PW1")
# motor.write_str("SA3")
# motor.write_str("PW0")
# time.sleep(3)

# print(motor.query_str("SA?"))