# Personalized Edge-AI Cardiac Digital Twin

A 24-hour biomedical hackathon project: a miniature "digital twin" of a
patient's cardiac state, built on an STM32 microcontroller and laptop
software.

## Problem

Build a personalized digital twin that predicts a patient's expected heart
rate / HRV from their activity, respiration, and SpO2 — and flags
deviations from *that patient's own baseline*, instead of using fixed
population thresholds (e.g. "HR > 100 bpm").

## Architecture

```
[Simulator]                [STM32]                [Laptop]
 patient state  --ECG-->   AD8232 + ADC  --UART-->  bridge/
 (vitals,          |       R-peak detect,             |
  scenarios)        --------> HR/HRV/temp             v
                                                    twin/ (predicts
                                                    expected HR/HRV,
                                                    personalizes,
                                                    flags residual
                                                    alerts)
                                                        |
                                                        v
                                                 dashboard/ (Streamlit:
                                                 live plots, alerts,
                                                 what-if sliders)
```

- **Simulator** (`simulator/`) plays the role of the "real patient,"
  generating a coupled physiology (activity, respiration, SpO2 → HR, HRV)
  and an ECG waveform from it.
- **Firmware** (`firmware/`) samples the analog ECG from the AD8232 on the
  STM32's ADC, filters it, detects R-peaks, computes HR/HRV, reads
  temperature, and streams feature lines over UART.
- **Bridge** (`bridge/`) reads the UART stream (or replays a recorded CSV)
  and republishes it in a shared Python dict format for the twin and
  dashboard.
- **Twin** (`twin/`) predicts each patient's *expected* HR/HRV from
  activity/respiration/SpO2, learns their personal baseline over the first
  1–2 minutes, and raises alerts based on the residual between predicted
  and measured values.
- **Dashboard** (`dashboard/`) is a Streamlit app showing the live
  waveform, twin-vs-measured comparison, alert panel, and "what-if"
  sliders (e.g. drop SpO2, raise activity) for a live demo.

See [`docs/protocol.md`](docs/protocol.md) for the exact data contract
between these components.

## Team

| Person | Owns | Folder |
|---|---|---|
| A | STM32 firmware and hardware | `firmware/` |
| B | Simulator, scenarios, corruption tests | `simulator/` |
| C | Twin, personalization, alerts, results | `twin/` |
| D | Repo, bridge, dashboard, pitch | `bridge/`, `dashboard/`, `docs/` |

## How to run

1. **Simulator only (no hardware):**
   ```
   pip install -r requirements.txt
   python simulator/run.py
   ```
2. **Dashboard (replay mode, no STM32 needed):**
   ```
   streamlit run dashboard/app.py -- --replay data/sample_ecg.csv
   ```
3. **Full pipeline (with STM32 connected):**
   - Flash `firmware/` via STM32CubeIDE + ST-LINK V2.
   - `python bridge/serial_reader.py --port <COM/tty port>`
   - `streamlit run dashboard/app.py`

## Repo structure

```
cardiac-digital-twin/
├── README.md            # problem, architecture, how to run
├── requirements.txt
├── .gitignore
├── docs/                # protocol.md (the contract), pitch notes
├── firmware/            # STM32CubeIDE project
├── simulator/           # patient state, ECG generator, scenarios
├── twin/                # model, personalization, residual alerts
├── bridge/              # serial reader, replay mode
├── dashboard/           # Streamlit app
└── data/                # sample ECG CSVs (small files only)
```

## Status

- [ ] STM32 → laptop data link working
- [ ] Personalized twin predicting expected HR/HRV
- [ ] Residual-based alerts
- [ ] Dashboard with at least one what-if demo
- [ ] (Nice-to-have) Sensor fusion layer for dropped sensors
- [ ] (Nice-to-have) On-device event-driven mode
