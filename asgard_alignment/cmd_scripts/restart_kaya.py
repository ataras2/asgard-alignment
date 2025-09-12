import asgard_alignment.controllino as co
import time


def main():
    cc = co.Controllino("192.168.100.10", init_motors=False)

    cc.turn_off("Kaya")
    time.sleep(2)
    cc.turn_on("Kaya")

    print("Kaya restarted")
