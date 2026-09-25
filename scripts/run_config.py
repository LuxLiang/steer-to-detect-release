"""Run a CLI with JSON defaults, followed by command-line overrides."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("entry", choices=["train", "evaluate", "evaluate_no_steer"])
    parser.add_argument("config", type=Path)
    args, extra = parser.parse_known_args()
    config = json.loads(args.config.read_text())
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / (args.entry + ".py"))]
    for key, value in config.items():
        if value is None or value is False:
            continue
        command.append("--" + key)
        if value is not True:
            command.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    command.extend(extra)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
