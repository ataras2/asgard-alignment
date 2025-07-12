import asgard_alignment.Engineering
import argparse

def main():
    parser = argparse.ArgumentParser(description="Autoalign Heimdallr beams.")
    parser.add_argument(
        "--shutter_pause_time",
        type=float,
        default=2.5,
        help="Seconds to pause after shuttering (default: 2.5)",
    )

    print(parser.parse_args())
    print(asgard_alignment.Engineering.get_matricies("c_red_one_focus"))