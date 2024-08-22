import pyvisa

print(pyvisa.ResourceManager().list_resources())

import asgard_alignment.NewportMotor as nm

motor = nm.LS16P('ASRL/dev/ttyACM0::INSTR', pyvisa.ResourceManager())
print(motor.write_str("OR"))