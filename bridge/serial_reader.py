"""
bridge/serial_reader.py

Reads feature lines from the STM32 (over UART) or replays a saved CSV,
and turns each one into the shared dict format everyone else's code
expects (see docs/protocol.md).

Two ways to run this file:

  1. Replay mode (no hardware needed, use this first):
       python serial_reader.py --replay ../data/sample_features.csv

  2. Live mode (once the STM32 is wired up and sending data):
       python serial_reader.py --port COM5
       (on Mac/Linux the port looks like /dev/ttyUSB0 or /dev/tty.usbserial-XXXX)

Either way, this script prints one parsed reading per line, and also
exposes a `read_features()` generator that other code (like the
dashboard) can loop over to get live readings.
"""

import argparse
import csv
import time


def parse_line(line):
    """
    Turn one raw line into a dict, or return None if the line
    isn't a valid feature line.

    Expected format (see docs/protocol.md):
        B,<timestamp_ms>,<hr_bpm>,<hrv_ms>,<temp_c>,<leads_off>
    Example:
        B,184532,72.4,45.1,36.8,0
    """
    line = line.strip()
    if not line.startswith("B,"):
        # Not a feature line (could be a debug line, blank line, etc.)
        return None

    parts = line.split(",")
    if len(parts) != 6:
        print(f"Skipping malformed line: {line}")
        return None

    try:
        return {
            "t": int(parts[1]) / 1000.0,      # convert ms -> seconds
            "hr_bpm": float(parts[2]),
            "hrv_ms": float(parts[3]),
            "temp_c": float(parts[4]),
            "leads_off": bool(int(parts[5])),
        }
    except ValueError:
        print(f"Skipping line with bad numbers: {line}")
        return None


def read_from_replay(csv_path, realtime=True):
    """
    Yield one feature dict per row of a CSV shaped like:
        timestamp_ms,hr_bpm,hrv_ms,temp_c,leads_off

    If realtime=True, sleeps between rows to mimic the real timing
    (based on the gap between timestamps), so a dashboard watching
    this looks like it's watching live data.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        last_t_ms = None
        for row in reader:
            t_ms = int(row["timestamp_ms"])

            if realtime and last_t_ms is not None:
                time.sleep((t_ms - last_t_ms) / 1000.0)
            last_t_ms = t_ms

            yield {
                "t": t_ms / 1000.0,
                "hr_bpm": float(row["hr_bpm"]),
                "hrv_ms": float(row["hrv_ms"]),
                "temp_c": float(row["temp_c"]),
                "leads_off": bool(int(row["leads_off"])),
            }


def read_from_serial(port, baudrate=115200):
    """
    Yield one feature dict per valid line read from the STM32 over UART.
    Requires: pip install pyserial
    """
    import serial  # imported here so replay mode works even without pyserial installed

    with serial.Serial(port, baudrate, timeout=1) as ser:
        print(f"Listening on {port} at {baudrate} baud...")
        while True:
            raw = ser.readline().decode(errors="ignore")
            if not raw:
                continue
            reading = parse_line(raw)
            if reading is not None:
                yield reading


def read_features(port=None, replay_csv=None, realtime=True):
    """
    The function other code (like the dashboard) should import and use.
    Pass either `port` (live) or `replay_csv` (replay), not both.
    """
    if replay_csv:
        yield from read_from_replay(replay_csv, realtime=realtime)
    elif port:
        yield from read_from_serial(port)
    else:
        raise ValueError("Must pass either port= or replay_csv=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read or replay STM32 feature data")
    parser.add_argument("--port", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--replay", help="Path to a CSV file to replay instead of live serial")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="In replay mode, play back as fast as possible instead of matching real timing",
    )
    args = parser.parse_args()

    if not args.port and not args.replay:
        parser.error("Provide either --port <serial port> or --replay <csv path>")

    for reading in read_features(
        port=args.port, replay_csv=args.replay, realtime=not args.fast
    ):
        print(reading)
