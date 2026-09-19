# Protocol: the shared data contract

Everyone codes against this file. If the format needs to change, propose it
at a sync point rather than changing it unilaterally — A, B, C, and D all
depend on it.

## 1. STM32 → laptop (UART line format)

The STM32 sends one line per feature update, comma-separated, prefixed
with `B` (for "beat"/"biofeature"):

```
B,<timestamp_ms>,<hr_bpm>,<hrv_ms>,<temp_c>,<leads_off>
```

| Field | Type | Notes |
|---|---|---|
| `timestamp_ms` | int | milliseconds since STM32 boot |
| `hr_bpm` | float | instantaneous heart rate from R-R interval |
| `hrv_ms` | float | short-window HRV (e.g. RMSSD over last N beats) |
| `temp_c` | float | temperature sensor reading, degrees C |
| `leads_off` | 0/1 | 1 if AD8232 LO+/LO- indicates a disconnected lead |

Example:
```
B,184532,72.4,45.1,36.8,0
```

Raw ADC/debug lines (if any) should be prefixed differently (e.g. `D,...`)
so the bridge can ignore them without breaking the parser.

## 2. Simulator → shared state (Python dict)

The simulator and the twin communicate using one shared dict per timestep:

```python
{
    "t": float,              # seconds since scenario start
    "activity": float,       # 0.0 (resting) to 1.0 (max exertion)
    "respiration_rate": float,   # breaths per minute
    "spo2": float,           # percent, e.g. 97.5
    "hr_true": float,        # ground-truth HR (simulator only, not sent to twin)
    "hrv_true": float,       # ground-truth HRV (simulator only)
    "ecg": list[float],      # ECG waveform samples for this timestep
    "scenario": str,         # e.g. "rest", "walk", "hypoxia", "arrhythmia"
}
```

The twin only ever sees `activity`, `respiration_rate`, `spo2`, and the
measured `hr_bpm`/`hrv_ms` coming from the bridge (or from the simulator's
`ecg` in early testing) — never `hr_true`/`hrv_true` directly, to avoid a
circular validation.

## 3. Bridge output (used by twin + dashboard)

The bridge (reading either live UART or a replayed CSV) republishes each
update as:

```python
{
    "t": float,
    "hr_bpm": float,
    "hrv_ms": float,
    "temp_c": float,
    "leads_off": bool,
    "activity": float,           # from simulator, if running in sim mode
    "respiration_rate": float,   # from simulator, if running in sim mode
    "spo2": float,                # from simulator, if running in sim mode
}
```

## 4. Twin output (used by dashboard)

```python
{
    "t": float,
    "hr_measured": float,
    "hr_expected": float,     # twin's prediction
    "residual": float,        # hr_measured - hr_expected
    "alert": bool,
    "alert_reason": str,      # e.g. "HR 18 bpm above personal baseline"
}
```

## 5. File conventions

- Sample ECG recordings go in `data/` as small CSVs: `t_ms,ecg_raw`.
- Each folder owner keeps their own internal helper files private to their
  folder; only the dicts/lines above are the public contract.

## Change log

- v1 — initial contract, drafted at kickoff.
