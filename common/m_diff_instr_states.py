#!/usr/bin/env python3
import json
import argparse
import re

def extract_relevant_positions(state_list, motor_prefixes, telescope_numbers):
    """
    Extract relevant motor positions given prefixes and telescope numbers from a list of entries.
    """
    result = {}
    telescope_numbers = set(map(str, telescope_numbers))
    for entry in state_list:
        name = entry.get("name", "")
        if any(name.startswith(prefix) for prefix in motor_prefixes):
            # Match telescope number at the end (e.g. BOTT2 → 2)
            match = re.search(r'(\d+)$', name)
            if match and match.group(1) in telescope_numbers:
                if entry.get("is_connected") and "position" in entry:
                    try:
                        result[name] = float(entry["position"])
                    except (ValueError, TypeError):
                        continue
    return result

def compute_position_differences(pos1, pos2):
    """
    Compute file1 - file2 differences for overlapping motors.
    """
    all_keys = sorted(set(pos1.keys()) | set(pos2.keys()))
    diffs = {}

    for key in all_keys:
        v1 = pos1.get(key)
        v2 = pos2.get(key)
        if v1 is not None and v2 is not None:
            diffs[key] = v1 - v2
        elif v1 is None:
            diffs[key] = f"Missing in file1: {v2}"
        elif v2 is None:
            diffs[key] = f"Missing in file2: {v1}"
    return diffs

def main():
    parser = argparse.ArgumentParser(description="Compare motor positions from two state files.")
    parser.add_argument("file1", help="Path to first JSON state file")
    parser.add_argument("file2", help="Path to second JSON state file")
    parser.add_argument("--motors", nargs="+", default=["BOTT", "BOTP", "BTT", "BTP"],
                        help="List of motor prefixes to consider (e.g. BOTT BOTP BTT BTP)")
    parser.add_argument("--telescopes", type=str, default="1,2,3,4",
                        help="Comma-separated telescope numbers to include (e.g. 1,2,4)")
    parser.add_argument("-o", "--output", default="motor_position_diff.json",
                        help="Output file for the differences")
    args = parser.parse_args()

    telescope_numbers = [int(n.strip()) for n in args.telescopes.split(",")]

    with open(args.file1) as f1, open(args.file2) as f2:
        state1 = json.load(f1)
        state2 = json.load(f2)

    pos1 = extract_relevant_positions(state1, args.motors, telescope_numbers)
    pos2 = extract_relevant_positions(state2, args.motors, telescope_numbers)

    diffs = compute_position_differences(pos1, pos2)

    with open(args.output, "w") as f:
        json.dump(diffs, f, indent=2)

    print(f"Written difference file to {args.output}")

if __name__ == "__main__":
    main()