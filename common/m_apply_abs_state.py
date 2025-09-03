import argparse
import json
import zmq

# apply a state to only subset of motors and/or telescopes 
## only tested on dry run
def load_state(file_path):
    with open(file_path) as f:
        return json.load(f)

def extract_motor_positions(state, motors, telescopes):
    positions = {}
    for entry in state:
        name = entry.get("name", "")
        if not entry.get("is_connected", False):
            continue
        for motor in motors:
            for tel in telescopes:
                beam = str(tel)
                if name == f"{motor}{beam}":
                    positions[(motor, beam)] = entry["position"]
    return positions

def send_absolute_moves(positions, mds_host, mds_port, dry_run=False):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{mds_host}:{mds_port}")
    for (motor, beam), pos in positions.items():
        cmd = f"moveabs {motor}{beam} {pos}"
        print(f"Sending: {cmd}")
        if not dry_run:
            socket.send_string(cmd)
            response = socket.recv_string()
            print(f"Response: {response}")
    socket.close()
    context.term()

def main():
    parser = argparse.ArgumentParser(description="Send absolute motor moves from a state.json file")
    parser.add_argument("state_json", help="Path to JSON file with motor states")
    parser.add_argument("--motors", nargs="+", default=["BOTT", "BOTP", "BTT", "BTP"],
                        help="List of motor prefixes to apply (default: BOTT BOTP BTT BTP)")
    parser.add_argument("--telescopes", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="List of telescope numbers (default: 1 2 3 4)")
    parser.add_argument("--mds_host", default="192.168.100.2", help="ZMQ host (default: 192.168.100.2)")
    parser.add_argument("--mds_port", type=int, default=5555, help="ZMQ port (default: 5555)")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without sending")
    args = parser.parse_args()

    state = load_state(args.state_json)
    positions = extract_motor_positions(state, args.motors, args.telescopes)
    send_absolute_moves(positions, args.mds_host, args.mds_port, dry_run=args.dry_run)

if __name__ == "__main__":
    main()