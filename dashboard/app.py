"""
dashboard/app.py

A simple Streamlit dashboard that shows live heart-rate data coming
from bridge/serial_reader.py (either a real STM32 over UART, or a
replayed CSV while there's no hardware connected yet).

How to run it:

  Replay mode (no hardware needed):
    streamlit run app.py -- --replay ../data/sample_features.csv

  Live mode (once the STM32 is wired up):
    streamlit run app.py -- --port COM5

Note the extra "--" before the real arguments — that's how you pass
arguments through Streamlit to this script.
"""

import argparse
import sys
import time

import streamlit as st

sys.path.append("../bridge")  # so we can import serial_reader.py from the sibling folder
from serial_reader import read_features


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--replay", help="Path to a CSV file to replay instead of live serial")
    parser.add_argument("--fast", action="store_true", help="Replay as fast as possible")
    # Streamlit passes its own args too, so we ignore ones we don't recognize
    args, _ = parser.parse_known_args()
    return args


st.set_page_config(page_title="Cardiac Digital Twin", layout="wide")
st.title("Cardiac Digital Twin — Live Monitor")

args = parse_args()

if not args.port and not args.replay:
    st.error(
        "No data source given. Run this with either:\n\n"
        "streamlit run app.py -- --replay ../data/sample_features.csv\n\n"
        "or\n\n"
        "streamlit run app.py -- --port COM5"
    )
    st.stop()

# --- Session state holds the rolling history of readings across reruns ---
if "history" not in st.session_state:
    st.session_state.history = {"t": [], "hr_bpm": [], "hrv_ms": [], "temp_c": []}
    st.session_state.feed = read_features(
        port=args.port, replay_csv=args.replay, realtime=not args.fast
    )

MAX_POINTS = 200  # keep only the most recent N readings on screen

# --- Layout: big HR number, alert box, then charts ---
col1, col2, col3 = st.columns(3)
hr_metric = col1.empty()
temp_metric = col2.empty()
alert_box = col3.empty()

hr_chart = st.empty()
hrv_chart = st.empty()

# --- Simple placeholder alert: flag if HR leaves a "normal" band ---
# This is a stand-in until the real twin (twin/) provides an
# expected-HR-based residual alert instead of a fixed threshold.
HR_LOW, HR_HIGH = 50, 120


def check_alert(hr):
    if hr < HR_LOW:
        return True, f"HR {hr:.0f} bpm is below {HR_LOW} (placeholder threshold)"
    if hr > HR_HIGH:
        return True, f"HR {hr:.0f} bpm is above {HR_HIGH} (placeholder threshold)"
    return False, "Normal"


# --- Main loop: pull one reading at a time and redraw ---
try:
    for reading in st.session_state.feed:
        h = st.session_state.history
        h["t"].append(reading["t"])
        h["hr_bpm"].append(reading["hr_bpm"])
        h["hrv_ms"].append(reading["hrv_ms"])
        h["temp_c"].append(reading["temp_c"])

        # trim to the last MAX_POINTS so the chart doesn't grow forever
        for key in h:
            h[key] = h[key][-MAX_POINTS:]

        hr_metric.metric("Heart Rate", f"{reading['hr_bpm']:.1f} bpm")
        temp_metric.metric("Temperature", f"{reading['temp_c']:.1f} °C")

        is_alert, reason = check_alert(reading["hr_bpm"])
        if is_alert:
            alert_box.error(f"⚠️ {reason}")
        else:
            alert_box.success("Normal")

        hr_chart.line_chart({"HR (bpm)": h["hr_bpm"]})
        hrv_chart.line_chart({"HRV (ms)": h["hrv_ms"]})

        if reading.get("leads_off"):
            st.warning("Leads off detected — check electrode contact")

except KeyboardInterrupt:
    pass
