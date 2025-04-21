from controllino import Controllino
import time
co = Controllino("172.16.8.200")
mod = 255
co.modulate("Lower Fan", mod)
co.modulate("Upper Fan", mod)
with open("log.txt", "a") as f:
	f.write("{}, {}, {}, {}, {}\n".format(time.asctime(), mod, mod, mod, mod))
	while(1):
		lower_T = co.analog_input('Lower T')
		upper_T = co.analog_input('Upper T')
		bench_T = co.analog_input('Bench T')
		floor_T = co.analog_input('Floor T')
		f.write("{}, {}, {}, {}, {}\n".format(time.asctime(), lower_T, upper_T, bench_T, floor_T))
		f.flush()
		time.sleep(5)
	
