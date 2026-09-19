"""
patient_simulator.py  -  ONE-FILE synthetic patient simulator  (Person B)

Everything Person B builds, in a single file:
  PART 1  physiology   hidden "true" patient (coupling rules, drift, per-patient parameters)
  PART 2  ECG          beat-by-beat ECG generator @ 250 Hz (PVC / AFib, wander, hum, motion)
  PART 3  scenarios    timelines, live what-if controls, get_frame(), ground-truth log
  PART 4  corruption   noise, dropouts, lost samples, electrode-off, rate mismatch
  PART 5  datasets     builds the CSVs for Person C into ./data
  PART 6  tests + CLI

Needs:  pip install numpy scipy pandas

COMMAND LINE (from the repo root, with this file inside simulator/)
  python simulator/patient_simulator.py test        # self-test, should end with ALL PASSED
  python simulator/patient_simulator.py demo        # print 60 s of the 3-minute demo scenario
  python simulator/patient_simulator.py datasets    # write the 20 CSVs into <repo root>/data
  (If the file sits at the repo root instead, drop the "simulator/" part.)

USE FROM PYTHON (scripts run from the repo root)
  from simulator.patient_simulator import Simulator, demo_3min, Corruptor

  sim = Simulator("average", demo_3min())   # "athlete" | "average" | "anxious"
  frame = sim.get_frame()                   # advances 1 s; shared dict + ecg + beats
  sim.set_state(spo2=86)                    # what-if slider (overrides the script)
  sim.clear_overrides()                     # back to the scripted timeline

The shared dict:  t (s), hr (bpm), rr (ms), hrv (ms RMSSD), temp (C), resp (/min),
spo2 (%), bp (mmHg systolic), activity (0..1), leads_ok (bool), source ("simulated").
Extra keys: ecg (250 samples = 1 s of mV), beats ([{t_ms, rr_ms, kind}]).
What the twin sees = MEASURED values. What is scored = TRUE values (sim.truth_log).
"""
import argparse
import os
import sys
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, lfilter


# ##########################################################################
# PART 1 - PHYSIOLOGY: the hidden true patient (was patient_state.py)
# ##########################################################################

# --------------------------------------------------------------------------
# Per-patient parameters (what makes each patient different)
# --------------------------------------------------------------------------
@dataclass
class PatientProfile:
    name: str = "average"
    hr0: float = 70.0          # resting heart rate (bpm)
    hrv0: float = 50.0         # resting HRV, RMSSD (ms)
    act_sens: float = 50.0     # HR rise (bpm) at activity = 1
    hypox_sens: float = 1.6    # HR rise (bpm) per SpO2 % below 95 (scaled, non-linear)
    spo2_base: float = 98.0    # normal SpO2 (%)
    bp_base: float = 120.0     # resting systolic BP (mmHg)
    resp_base: float = 14.0    # resting breaths / min
    temp_base: float = 36.7    # body temp (deg C)
    fatigue: float = 0.15      # extra HR creep during sustained exertion
    noise: float = 1.0         # multiplier on sensor noise
    seed: int = 0


PROFILES = {
    "athlete": PatientProfile("athlete", hr0=55, hrv0=70, act_sens=35, hypox_sens=1.2,
                              spo2_base=99, bp_base=110, resp_base=12, temp_base=36.5,
                              fatigue=0.08, seed=1),
    "average": PatientProfile("average", hr0=70, hrv0=50, act_sens=50, hypox_sens=1.6,
                              spo2_base=98, bp_base=120, resp_base=14, temp_base=36.7,
                              fatigue=0.15, seed=2),
    "anxious": PatientProfile("anxious", hr0=82, hrv0=25, act_sens=55, hypox_sens=2.0,
                              spo2_base=97, bp_base=135, resp_base=16, temp_base=36.9,
                              fatigue=0.22, seed=3),
}


def random_profile(seed: int, name=None) -> PatientProfile:
    """A new synthetic patient with plausible, randomly different parameters."""
    r = np.random.default_rng(1000 + seed)
    return PatientProfile(
        name=name or f"patient_{seed:02d}",
        hr0=float(r.uniform(56, 86)), hrv0=float(r.uniform(22, 72)),
        act_sens=float(r.uniform(32, 60)), hypox_sens=float(r.uniform(1.1, 2.2)),
        spo2_base=float(r.uniform(96.5, 99)), bp_base=float(r.uniform(106, 136)),
        resp_base=float(r.uniform(12, 17)), temp_base=float(r.uniform(36.4, 37.0)),
        fatigue=float(r.uniform(0.06, 0.25)), seed=seed)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def approach(x, target, tau, dt=1.0):
    """First-order lag: move x toward target with time constant tau (s)."""
    return x + (target - x) * (1.0 - np.exp(-dt / tau))


class OU:
    """Slow mean-reverting random drift (Ornstein-Uhlenbeck)."""

    def __init__(self, tau, sigma):
        self.tau, self.sigma, self.x = tau, sigma, 0.0

    def step(self, z, dt=1.0):
        self.x += -self.x / self.tau * dt + self.sigma * np.sqrt(dt) * z
        return self.x


ARRHYTHMIAS = (None, "pvc", "afib", "tachy", "brady")


# --------------------------------------------------------------------------
# The hidden patient
# --------------------------------------------------------------------------
class PatientState:
    """Advance with step(); returns the TRUE state dict for that second.

    step() always draws the same number of random numbers, so two patients
    created with the same seed stay noise-synchronised (used for the
    'shadow' patient that shows what HR would have been without an anomaly).
    """

    def __init__(self, profile: PatientProfile):
        self.p = profile
        self.rng = np.random.default_rng(profile.seed)
        self.t = 0.0
        self.act = 0.0
        self.resp = profile.resp_base
        self.spo2 = profile.spo2_base
        self.hr = profile.hr0
        self.sbp = profile.bp_base
        self.temp = profile.temp_base
        self.fatigue_state = 0.0
        self.hrv = profile.hrv0
        self.d_hr = OU(tau=90, sigma=0.6)      # slow HR drift (bpm)
        self.d_resp = OU(tau=60, sigma=0.15)
        self.d_bp = OU(tau=120, sigma=0.5)
        self.d_temp = OU(tau=300, sigma=0.01)
        self.d_spo2 = OU(tau=60, sigma=0.08)

    # ------------------------------------------------------------------
    def step(self, act_target=0.0, hypoxia=0.0, sleep=False, arrhythmia=None,
             bp_offset=0.0, dt=1.0):
        """
        act_target : 0..1   (0 rest, ~0.5 walking, 1 running)
        hypoxia    : 0..1   severity, 1 -> SpO2 target ~ 14 points below baseline
        sleep      : bool
        arrhythmia : None | 'pvc' | 'afib' | 'tachy' | 'brady'
        bp_offset  : mmHg added to the BP target (what-if hypertension / hypotension)
        """
        p = self.p
        z = self.rng.standard_normal(8)          # fixed draw count (shadow-safe)
        act_target = float(np.clip(act_target, 0, 1))
        hypoxia = float(np.clip(hypoxia, 0, 1))
        if sleep:
            act_target = 0.0

        # --- activity (the person's effort), quick response
        self.act = approach(self.act, act_target, 4.0, dt)

        # --- fatigue: builds during sustained effort, decays slowly at rest
        self.fatigue_state = approach(self.fatigue_state, self.act ** 2, 90.0, dt)

        # --- SpO2: falls toward hypoxic target, mild dip with hard exertion
        spo2_target = p.spo2_base - 1.2 * self.act - 14.0 * hypoxia
        tau = 8.0 if hypoxia > 0 else 15.0
        self.spo2 = approach(self.spo2, spo2_target, tau, dt)
        spo2_true = float(np.clip(self.spo2 + self.d_spo2.step(z[4], dt), 60, 100))

        # --- respiration: exertion, hypoxic drive, sleep slows it
        resp_target = p.resp_base + 16.0 * self.act ** 0.9
        resp_target += 0.5 * max(0.0, 95.0 - spo2_true)          # hypoxic ventilatory drive
        if sleep:
            resp_target -= 2.0
        self.resp = approach(self.resp, resp_target, 7.0, dt)
        resp_true = float(np.clip(self.resp + self.d_resp.step(z[1], dt), 6, 45))

        # --- heart rate (non-linear on purpose)
        hyp_drive = p.hypox_sens * (max(0.0, 95.0 - spo2_true) ** 1.15)
        hr_target = (p.hr0
                     + p.act_sens * self.act ** 0.8              # concave activity response
                     + p.fatigue * 30.0 * self.fatigue_state     # exertion creep
                     + hyp_drive)
        if sleep:
            hr_target -= 0.12 * p.hr0
        if arrhythmia == "tachy":
            hr_target += 45
        elif arrhythmia == "brady":
            hr_target -= 22
        elif arrhythmia == "afib":
            hr_target += 28
        hr_target += self.d_hr.step(z[0], dt)
        tau_hr = 5.0 if hr_target > self.hr else 11.0             # recovers slower than it rises
        self.hr = approach(self.hr, hr_target, tau_hr, dt)
        hr_true = float(np.clip(self.hr, 35, 195))

        # --- HRV (RMSSD, ms): exertion lowers it, slow breathing raises it (RSA)
        hrv_target = p.hrv0 * (1.0 - 0.6 * self.act ** 0.8)
        hrv_target *= 1.0 + 0.025 * (p.resp_base - resp_true)
        hrv_target *= 1.0 - 0.012 * max(0.0, 95.0 - spo2_true)
        if sleep:
            hrv_target *= 1.25
        if arrhythmia == "afib":
            hrv_target = 105.0 + 8 * z[5]
        elif arrhythmia == "tachy":
            hrv_target *= 0.6
        self.hrv = approach(self.hrv, hrv_target, 8.0, dt)
        hrv_true = float(np.clip(self.hrv, 5, 160))

        # --- blood pressure (systolic) and temperature
        bp_target = p.bp_base + 30.0 * self.act + bp_offset - (6.0 if sleep else 0.0)
        self.sbp = approach(self.sbp, bp_target, 12.0, dt)
        bp_true = float(self.sbp + self.d_bp.step(z[2], dt))
        temp_target = p.temp_base + 0.5 * self.act - (0.25 if sleep else 0.0)
        self.temp = approach(self.temp, temp_target, 150.0, dt)
        temp_true = float(self.temp + self.d_temp.step(z[3], dt))

        self.t += dt
        return dict(t=self.t, hr=hr_true, rr=60000.0 / hr_true, hrv=hrv_true,
                    temp=temp_true, resp=resp_true, spo2=spo2_true, bp=bp_true,
                    activity=float(self.act), arrhythmia=arrhythmia or "none",
                    hypoxia=hypoxia, sleep=bool(sleep))


# ##########################################################################
# PART 2 - ECG GENERATOR (was ecg_sim.py)
# ##########################################################################

class ECGSynth:
    def __init__(self, fs=250, seed=0, noise=1.0, mains_hz=50.0):
        self.fs = fs
        self.rng = np.random.default_rng(seed + 12345)
        self.noise = noise
        self.mains_hz = mains_hz
        self.t = 0.0                    # time rendered so far (s)
        self.next_r = 0.35              # time of the next R peak (s)
        self.beats = []                 # pending / recent beats {t, rr, kind}
        self._rsa_phase = 0.0
        self._forced_rr = None          # used for PVC compensatory pause
        self._resp_phase = 0.0          # for respiration-linked baseline wander
        self._motion_zi = np.zeros(1)   # filter state for motion artifact
        self.last_rr = 60000 / 70

    # ------------------------------------------------------------------
    # RR interval generation
    # ------------------------------------------------------------------
    def _next_rr(self, hr, hrv, resp, rhythm):
        """Return (rr_ms_from_previous_beat_to_next, kind_of_next_beat)."""
        base = 60000.0 / max(hr, 30.0)

        if self._forced_rr is not None:                 # compensatory pause after PVC
            rr, self._forced_rr = self._forced_rr, None
            return rr, "normal"

        if rhythm == "afib":                            # irregularly irregular
            s = (hrv / np.sqrt(2)) / max(base, 1.0)
            rr = base * np.exp(self.rng.normal(0, np.clip(s, 0.08, 0.35)))
            return float(np.clip(rr, 280, 1600)), "afib"

        # normal sinus rhythm with RSA + jitter, calibrated to RMSSD ~ hrv
        f = resp / 60.0
        k = max(np.sqrt(2) * abs(np.sin(np.pi * f * base / 1000.0)), 0.35)
        rsa_amp = (hrv * np.sqrt(0.6)) / k
        jit_sd = (hrv * np.sqrt(0.4)) / np.sqrt(2)
        self._rsa_phase += 2 * np.pi * f * base / 1000.0
        rr = base + rsa_amp * np.sin(self._rsa_phase) + self.rng.normal(0, jit_sd)
        rr = float(np.clip(rr, 0.55 * base, 1.6 * base))

        if rhythm == "pvc" and self.rng.random() < 0.10:
            self._forced_rr = max(2 * base - 0.62 * base, 0.6 * base)   # pause after PVC
            return 0.62 * base, "pvc"                                   # premature beat
        return rr, "normal"

    # ------------------------------------------------------------------
    # beat templates
    # ------------------------------------------------------------------
    @staticmethod
    def _beat(tau, rr_s, kind):
        """ECG contribution (mV) of one beat at times tau (s) relative to its R peak."""
        g = lambda mu, a, s: a * np.exp(-0.5 * ((tau - mu) / s) ** 2)
        sq = np.sqrt(rr_s)
        if kind == "pvc":
            return (g(0.0, 1.25, 0.028) + g(0.055, -0.45, 0.030)
                    + g(0.30 * sq, -0.45, 0.065))
        y = (g(0.0, 1.00, 0.010) + g(-0.036, -0.10, 0.008) + g(0.030, -0.22, 0.009)
             + g(0.27 * sq, 0.30, 0.045 * sq))
        if kind != "afib":
            y = y + g(-0.17 * sq, 0.12, 0.022)          # P wave (absent in AFib)
        return y

    # ------------------------------------------------------------------
    def render(self, dur, hr, hrv, resp, rhythm=None, activity=0.0, lead_off=False):
        """
        Advance the ECG by `dur` seconds.
        Returns (ecg_mv: ndarray[int(dur*fs)], beats: list of new beat dicts
                 {t_ms, rr_ms, kind} whose R peak fell inside this chunk).
        """
        fs, n = self.fs, int(round(dur * self.fs))
        t0, t1 = self.t, self.t + dur
        # schedule beats a little beyond the chunk so T waves are complete
        while self.next_r < t1 + 0.8:
            rr_ms, kind = self._next_rr(hr, hrv, resp, rhythm)
            self.beats.append(dict(t=self.next_r, rr=rr_ms, kind=kind))
            self.next_r += rr_ms / 1000.0
        ts = t0 + np.arange(n) / fs
        x = np.zeros(n)
        new_beats = []
        prev_t = None
        for b in self.beats:
            if t0 - 0.7 <= b["t"] < t1 + 0.7:
                x += self._beat(ts - b["t"], max(b["rr"], 400) / 1000.0, b["kind"])
            if t0 <= b["t"] < t1:
                new_beats.append(dict(t_ms=b["t"] * 1000.0, rr_ms=b["rr"], kind=b["kind"]))
        self.beats = [b for b in self.beats if b["t"] > t0 - 1.0]

        if rhythm == "afib":                                # fibrillatory f-waves
            x += 0.03 * np.sin(2 * np.pi * 6.0 * ts + 1.3) * (0.6 + 0.4 * self.rng.random())

        # ---- baseline wander (slow + respiration-linked)
        f_resp = resp / 60.0
        ph = self._resp_phase + 2 * np.pi * f_resp * np.arange(n) / fs
        self._resp_phase = float(ph[-1] + 2 * np.pi * f_resp / fs) % (2 * np.pi)
        x += 0.10 * np.sin(2 * np.pi * 0.17 * ts + 0.7) + 0.05 * np.sin(ph)

        # ---- noise: white + mains hum + motion artifact (grows with activity)
        x += self.rng.normal(0, 0.015 * self.noise, n)
        x += 0.01 * self.noise * np.sin(2 * np.pi * self.mains_hz * ts)
        if activity > 0.05:
            w = self.rng.normal(0, 1, n)
            slow, self._motion_zi = lfilter([0.02], [1, -0.98], w, zi=self._motion_zi)
            x += (3.0 * activity ** 2) * slow
            if activity > 0.4 and self.rng.random() < 0.25 * activity:   # occasional spike burst
                i = self.rng.integers(0, max(n - 20, 1))
                x[i:i + 20] += self.rng.normal(0, 0.25 * activity, min(20, n - i))

        if lead_off:                                        # electrode disconnected: flat rail + hum
            x = 0.02 * np.sin(2 * np.pi * self.mains_hz * ts) + 0.005 * self.rng.standard_normal(n) + 1.4

        self.t = t1
        if new_beats:
            self.last_rr = new_beats[-1]["rr_ms"]
        return x, new_beats


def to_adc_counts(ecg_mv, bits=12, gain=100.0, vref=3.3):
    """
    Convert mV -> STM32 ADC counts the way an AD8232 module would present it:
    output = 1.5 V mid-rail + gain * ECG. (Default AD8232 board gain is about 100.)
    Use ONLY for the stretch goal of streaming simulated ECG to the STM32.
    """
    volts = 1.5 + np.asarray(ecg_mv) * 1e-3 * gain
    counts = np.round(volts / vref * (2 ** bits - 1))
    return np.clip(counts, 0, 2 ** bits - 1).astype(int)


# ##########################################################################
# PART 3 - SCENARIOS + get_frame() (was scenarios.py)
# ##########################################################################

SHARED_KEYS = ["t", "hr", "rr", "hrv", "temp", "resp", "spo2", "bp",
               "activity", "leads_ok", "source"]


# ==========================================================================
# Timeline
# ==========================================================================
@dataclass
class Segment:
    start: float
    end: float
    label: str
    activity: float = 0.0          # 0 rest, 0.5 walking, 1 running
    hypoxia: float = 0.0           # severity 0..1
    sleep: bool = False
    arrhythmia: str = None         # None | pvc | afib | tachy | brady
    lead_off: bool = False
    bp_offset: float = 0.0


class Timeline:
    """Ordered list of segments. Build with .add(label, seconds, **params)."""

    def __init__(self, name="custom"):
        self.name, self.segments, self._t = name, [], 0.0

    def add(self, label, seconds, **params):
        self.segments.append(Segment(self._t, self._t + seconds, label, **params))
        self._t += seconds
        return self

    @property
    def duration(self):
        return self._t

    def at(self, t):
        for s in self.segments:
            if s.start <= t < s.end:
                return s
        return Segment(t, t + 1, "rest")            # after the script ends: rest

    def table(self):
        return [(s.start, s.end, s.label) for s in self.segments]


# ---- ready-made scripts --------------------------------------------------
def demo_5min():
    """Full 5-minute script used for datasets and the 'done when' check."""
    return (Timeline("demo_5min")
            .add("rest", 60)
            .add("walking", 60, activity=0.5)
            .add("recovery", 30)
            .add("hypoxia", 60, hypoxia=0.9)
            .add("recovery", 30)
            .add("pvc", 30, arrhythmia="pvc")
            .add("afib", 20, arrhythmia="afib")
            .add("recovery", 10))


def demo_3min():
    """The 3-minute LIVE DEMO script (matches the pitch: rest -> walk -> hypoxia).
    Timings are exact so D can rehearse against them:
      0:00-0:45 rest | 0:45-1:30 walking | 1:30-1:45 recovery
      1:45-2:30 hypoxia | 2:30-3:00 recovery
    """
    return (Timeline("demo_3min")
            .add("rest", 45)
            .add("walking", 45, activity=0.5)
            .add("recovery", 15)
            .add("hypoxia", 45, hypoxia=0.9)
            .add("recovery", 30))


def sleep_demo():
    return (Timeline("sleep_demo").add("rest", 60).add("sleep", 180, sleep=True)
            .add("waking", 60))


def training_random(minutes=20, seed=0):
    """Healthy data ONLY (no anomalies) with varied activity - for fitting the twin."""
    r = np.random.default_rng(seed)
    tl, total = Timeline(f"train_{seed}"), 0.0
    levels = [0.0, 0.0, 0.15, 0.3, 0.5, 0.7, 0.9]
    while total < minutes * 60:
        d = float(r.integers(25, 70))
        if r.random() < 0.10:
            tl.add("train_sleep", d, sleep=True)
        else:
            tl.add("train", d, activity=float(r.choice(levels)))
        total += d
    return tl


# ==========================================================================
# Simulator
# ==========================================================================
class Simulator:
    def __init__(self, profile="average", timeline=None, fs=250, noise=1.0,
                 source="simulated", shadow=True, seed=None):
        if isinstance(profile, str):
            profile = PROFILES[profile]
        if seed is not None:
            profile = PatientProfile(**{**profile.__dict__, "seed": seed})
        self.profile, self.fs, self.source = profile, fs, source
        self.timeline = timeline or demo_5min()
        self.state = PatientState(profile)
        # 'shadow' = same patient, same random noise, but NO hypoxia/arrhythmia.
        # Lets C score residual alerts against the true "what HR should have been".
        self.shadow = PatientState(profile) if shadow else None
        self.ecg = ECGSynth(fs=fs, seed=profile.seed, noise=noise * profile.noise)
        self.rng = np.random.default_rng(profile.seed + 777)
        self.overrides = {}
        self.truth_log = []
        self._rr_hist = deque(maxlen=30)
        self._last_meas = dict(hr=profile.hr0, rr=60000 / profile.hr0, hrv=profile.hrv0)
        self.t = 0.0

    # ---------------------------------------------------------------- live controls
    def set_state(self, activity=None, spo2=None, hypoxia=None, arrhythmia="__keep__",
                  sleep=None, lead_off=None, bp_offset=None):
        """What-if controls (D's sliders call this). Any arg left as None is unchanged.
        spo2: target SpO2 (%) -> converted to a hypoxia severity so the physiology stays coherent.
        arrhythmia: None/'none' clears it, or one of pvc | afib | tachy | brady."""
        o = self.overrides
        if activity is not None:
            o["activity"] = float(np.clip(activity, 0, 1))
        if hypoxia is not None:
            o["hypoxia"] = float(np.clip(hypoxia, 0, 1))
        if spo2 is not None:
            act = o.get("activity", self.timeline.at(self.t).activity)
            sev = (self.profile.spo2_base - 1.2 * act - float(spo2)) / 14.0
            o["hypoxia"] = float(np.clip(sev, 0, 1))
        if arrhythmia != "__keep__":
            a = None if arrhythmia in (None, "none") else arrhythmia
            if a not in ARRHYTHMIAS:
                raise ValueError(f"arrhythmia must be one of {ARRHYTHMIAS}")
            o["arrhythmia"] = a
        if sleep is not None:
            o["sleep"] = bool(sleep)
        if lead_off is not None:
            o["lead_off"] = bool(lead_off)
        if bp_offset is not None:
            o["bp_offset"] = float(bp_offset)

    def set_activity(self, x): self.set_state(activity=x)
    def inject_hypoxia(self, severity=0.9): self.set_state(hypoxia=severity)
    def clear_hypoxia(self): self.set_state(hypoxia=0.0)
    def set_arrhythmia(self, kind): self.set_state(arrhythmia=kind)
    def clear_arrhythmia(self): self.set_state(arrhythmia=None)
    def set_lead_off(self, on=True): self.set_state(lead_off=on)

    def clear_overrides(self, *names):
        """No args = back to the scripted timeline. Or clear only e.g. 'hypoxia'."""
        if names:
            for n in names:
                self.overrides.pop(n, None)
        else:
            self.overrides.clear()

    def _current_inputs(self):
        seg = self.timeline.at(self.t)
        p = dict(activity=seg.activity, hypoxia=seg.hypoxia, sleep=seg.sleep,
                 arrhythmia=seg.arrhythmia, lead_off=seg.lead_off,
                 bp_offset=seg.bp_offset)
        label = seg.label
        p.update(self.overrides)
        if self.overrides:
            label = label + "+live"
        return p, label

    # ---------------------------------------------------------------- main call
    def get_frame(self, with_truth=False, with_ecg=True):
        """Advance 1 second. Returns the shared dict (+ ecg, beats)."""
        p, label = self._current_inputs()
        truth = self.state.step(p["activity"], p["hypoxia"], p["sleep"],
                                p["arrhythmia"], p["bp_offset"])
        normal = None
        if self.shadow is not None:
            normal = self.shadow.step(p["activity"], 0.0, p["sleep"], None, p["bp_offset"])

        # ECG for this second, driven by the TRUE state
        ecg, beats = self.ecg.render(1.0, truth["hr"], truth["hrv"], truth["resp"],
                                     p["arrhythmia"], truth["activity"], p["lead_off"])
        if p["lead_off"]:
            beats = []                       # electrode off: the detector sees no beats
        for b in beats:
            self._rr_hist.append(b["rr_ms"])

        # ---- what an STM32 would MEASURE from that ECG
        if p["lead_off"] or len(self._rr_hist) < 3:
            meas = dict(self._last_meas) if p["lead_off"] else dict(
                hr=truth["hr"], rr=truth["rr"], hrv=truth["hrv"])
        else:
            rr = np.array(self._rr_hist)
            rr_mean = float(rr[-5:].mean())
            meas = dict(hr=60000.0 / rr_mean, rr=rr_mean,
                        hrv=float(np.sqrt(np.mean(np.diff(rr) ** 2))))
        if not p["lead_off"]:
            self._last_meas = meas

        n = self.profile.noise
        r = self.rng
        frame = dict(
            t=truth["t"], hr=round(meas["hr"], 1), rr=round(meas["rr"], 1),
            hrv=round(meas["hrv"], 1),
            temp=round(truth["temp"] + r.normal(0, 0.03 * n), 2),
            resp=round(truth["resp"] + r.normal(0, 0.5 * n), 1),
            spo2=round(float(np.clip(truth["spo2"] + r.normal(0, 0.4 * n), 0, 100)), 1),
            bp=round(truth["bp"] + r.normal(0, 2.0 * n), 1),
            activity=round(float(np.clip(truth["activity"] + r.normal(0, 0.02 * n), 0, 1)), 3),
            leads_ok=not p["lead_off"], source=self.source)

        # ---- ground-truth log (for scoring the twin)
        rec = dict(truth)
        rec.update(label=label, lead_off=p["lead_off"],
                   anomaly=int(truth["hypoxia"] > 0.05 or truth["arrhythmia"] != "none"))
        if normal is not None:
            rec.update(hr_normal=normal["hr"], hrv_normal=normal["hrv"],
                       spo2_normal=normal["spo2"], resp_normal=normal["resp"],
                       hr_residual_true=truth["hr"] - normal["hr"])
        self.truth_log.append(rec)

        self.t = truth["t"]
        if with_ecg:
            frame["ecg"], frame["beats"] = ecg, beats
        if with_truth:
            frame["truth"] = rec
        return frame

    # ---------------------------------------------------------------- helpers
    def run(self, seconds=None, **kw):
        """Generator of frames until the timeline (or `seconds`) is done."""
        n = int(seconds if seconds is not None else self.timeline.duration)
        for _ in range(n):
            yield self.get_frame(**kw)

    def collect(self, seconds=None, keep_ecg=False):
        """Run and return (measured_df, truth_df, ecg_array_or_None)."""
        rows, ecgs = [], []
        for f in self.run(seconds, with_ecg=True):
            ecgs.append(f["ecg"])
            rows.append({k: f[k] for k in SHARED_KEYS})
        return (pd.DataFrame(rows), pd.DataFrame(self.truth_log),
                np.concatenate(ecgs) if keep_ecg else None)

    def truth_df(self):
        return pd.DataFrame(self.truth_log)

    def save_truth(self, path):
        self.truth_df().round(3).to_csv(path, index=False)
        return path

    @staticmethod
    def b_lines(frame):
        """STM32-style per-heartbeat lines for D's replay mode (docs/protocol.md):
        B,<t_ms>,<rr_ms>,<hr_bpm>,<hrv_ms>,<temp_c>,<leads_ok>"""
        return [f"B,{int(b['t_ms'])},{int(round(b['rr_ms']))},{frame['hr']:.0f},"
                f"{frame['hrv']:.0f},{frame['temp']:.1f},{int(frame['leads_ok'])}"
                for b in frame.get("beats", [])]


# ##########################################################################
# PART 4 - CORRUPTION INJECTOR (was corrupt.py)
# ##########################################################################

VITAL_FIELDS = ["hr", "rr", "hrv", "temp", "resp", "spo2", "bp", "activity"]


class Corruptor:
    def __init__(self, level=0.2, seed=0):
        self.level = float(np.clip(level, 0.0, 0.5))
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ ECG
    def ecg_corrupt(self, ecg, fs=250, mode="nan"):
        """
        ecg  : 1-D clean ECG (mV)
        mode : "nan"    -> lost samples stay in place as NaN (easy to align)
               "remove" -> lost samples are deleted; returned time axis has holes

        Returns dict:
            t        time axis (s)               (holes if mode == "remove")
            x        corrupted ECG (mV)
            valid    bool mask on the ORIGINAL timeline: True = sample survived
            leads_ok bool mask on the ORIGINAL timeline: False = electrode off
            clean    the untouched input (ground truth)
        """
        rng, lv = self.rng, self.level
        x = np.array(ecg, dtype=float)
        n = len(x)
        dur = n / fs
        t_full = np.arange(n) / fs
        leads_ok = np.ones(n, bool)
        valid = np.ones(n, bool)
        if lv == 0:
            return dict(t=t_full, x=x, valid=valid, leads_ok=leads_ok, clean=np.array(ecg))

        # 1) broadband noise grows with level
        x += rng.normal(0, 0.02 + 0.35 * lv, n)

        # 2) motion-artifact bursts: slow big swings + jitter, 0.5-2 s each
        n_bursts = rng.poisson(lv * dur / 6.0)
        for _ in range(n_bursts):
            L_ = int(rng.uniform(0.5, 2.0) * fs)
            i = int(rng.integers(0, max(n - L_, 1)))
            tt = np.arange(min(L_, n - i)) / fs
            amp = rng.uniform(0.4, 1.4)
            x[i:i + len(tt)] += (amp * np.sin(2 * np.pi * rng.uniform(0.6, 2.5) * tt + rng.uniform(0, 6.28))
                                 + rng.normal(0, 0.15 * amp, len(tt)))

        # 3) electrode disconnects: flat line, leads_ok False (20% of the loss budget)
        off_budget = int(0.20 * lv * n)
        while (~leads_ok).sum() < off_budget:
            L_ = int(min(off_budget - (~leads_ok).sum(), rng.uniform(0.5, 3.0) * fs))
            L_ = max(L_, 1)
            i = int(rng.integers(0, max(n - L_, 1)))
            x[i:i + L_] = 1.4 + rng.normal(0, 0.003, len(x[i:i + L_]))     # railed baseline
            leads_ok[i:i + L_] = False

        # 4) lost samples (80% of the budget): 40% random singles, 60% burst gaps.
        #    Loops until the budget is met exactly, so level 0.5 really loses ~40% + 10% off.
        loss_budget = int(0.80 * lv * n)
        singles = int(0.4 * loss_budget)
        if singles:
            valid[rng.choice(n, singles, replace=False)] = False
        while (~valid).sum() < loss_budget:
            L_ = int(min(loss_budget - (~valid).sum(), rng.uniform(0.1, 1.5) * fs))
            L_ = max(L_, 1)
            i = int(rng.integers(0, max(n - L_, 1)))
            valid[i:i + L_] = False

        if mode == "nan":
            out = np.where(valid, x, np.nan)
            return dict(t=t_full, x=out, valid=valid, leads_ok=leads_ok, clean=np.array(ecg))
        return dict(t=t_full[valid], x=x[valid], valid=valid, leads_ok=leads_ok,
                    clean=np.array(ecg))

    # --------------------------------------------------------- vitals frames
    def frame_corrupt(self, frame):
        """Corrupt ONE shared-dict frame. Returns a copy (clean frame untouched).
        Each vital independently may: drop out (None), spike, or repeat a stale value.
        Probability per field per frame = level. If the leads are off, hr/rr/hrv are None."""
        f = dict(frame)
        rng = self.rng
        for k in VITAL_FIELDS:
            if k not in f or f[k] is None:
                continue
            if rng.random() < self.level:
                kind = rng.choice(["drop", "spike", "stale"], p=[0.5, 0.3, 0.2])
                if kind == "drop":
                    f[k] = None
                elif kind == "spike":
                    f[k] = float(f[k]) * float(rng.choice([0.4, 0.6, 1.6, 2.2]))
                else:
                    f[k] = getattr(self, "_stale", {}).get(k, f[k])
        # leads-off flag flips occasionally at high corruption
        if rng.random() < self.level * 0.3:
            f["leads_ok"] = False
            for k in ("hr", "rr", "hrv"):
                f[k] = None
        self._stale = {k: v for k, v in frame.items() if k in VITAL_FIELDS}
        return f

    def stream(self, frames):
        """Wrap a frame generator: yields corrupted copies (and skips whole frames at random)."""
        for fr in frames:
            if self.rng.random() < 0.3 * self.level:       # entire second lost
                continue
            yield self.frame_corrupt(fr)

    def df_corrupt(self, df):
        """Corrupt a measured DataFrame (columns = shared dict). Returns a new DataFrame
        with NaN where data is missing; the input is untouched."""
        rows = [self.frame_corrupt(r) for r in df.to_dict("records")]
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ rate mismatch
def multirate(df, rates=None, jitter=0.0, seed=0):
    """
    Turn a 1 Hz DataFrame into signals that arrive at DIFFERENT rates (problem #3).

    rates: dict field -> Hz, e.g. {"temp": 0.2, "spo2": 0.5, "bp": 0.1, "resp": 0.5}
           (anything not listed stays at the frame rate of `df`)
    jitter: std (s) of random timestamp jitter (late / early arrival)

    Returns dict field -> DataFrame[t, value] with only the samples that "arrived".
    The fusion layer must align these onto one timeline (hold / interpolate / Kalman).
    """
    rates = rates or {"temp": 0.2, "spo2": 0.5, "bp": 0.1, "resp": 0.5}
    rng = np.random.default_rng(seed)
    base_hz = 1.0 / float(np.median(np.diff(df["t"].values)))
    out = {}
    for k in VITAL_FIELDS:
        if k not in df:
            continue
        step = max(int(round(base_hz / rates.get(k, base_hz))), 1)
        sub = df.iloc[::step][["t", k]].copy()
        if jitter > 0:
            sub["t"] = sub["t"] + rng.normal(0, jitter, len(sub))
        out[k] = sub.rename(columns={k: "value"}).reset_index(drop=True)
    return out


def hold_last(series_df, t_grid):
    """Baseline for the fusion score: sample-and-hold a slow signal onto a fast grid."""
    s = series_df.sort_values("t")
    idx = np.searchsorted(s["t"].values, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(s) - 1)
    return s["value"].values[idx]


def rmse(estimate, truth):
    """RMSE ignoring NaNs - use for 'error vs corruption level' curves."""
    e, t = np.asarray(estimate, float), np.asarray(truth, float)
    m = ~(np.isnan(e) | np.isnan(t))
    return float(np.sqrt(np.mean((e[m] - t[m]) ** 2))) if m.any() else float("nan")


# ##########################################################################
# PART 5 - DATASET GENERATOR (was generate_datasets.py)
# ##########################################################################

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "simulator" else _HERE
OUT = os.path.join(_ROOT, "data")          # <repo root>/data


def reseed(profile, seed):
    return PatientProfile(**{**profile.__dict__, "seed": seed})


def generate_datasets(out=OUT):
    os.makedirs(out, exist_ok=True)
    patients = [PROFILES["athlete"], PROFILES["average"], PROFILES["anxious"],
                random_profile(4), random_profile(5)]
    written = []

    for prof in patients:
        # --- training set: healthy only, different noise realisation than the demo
        sim = Simulator(reseed(prof, prof.seed + 100), training_random(20, seed=prof.seed))
        meas, truth, _ = sim.collect()
        meas.round(3).to_csv(f"{out}/{prof.name}_train.csv", index=False)
        assert truth["anomaly"].sum() == 0            # training data must be anomaly-free

        # --- demo scenario: measured (twin input) + truth (scoring)
        sim = Simulator(prof, demo_5min())
        meas, truth, ecg = sim.collect(keep_ecg=True)
        meas.round(3).to_csv(f"{out}/{prof.name}_demo.csv", index=False)
        sim.save_truth(f"{out}/{prof.name}_demo_truth.csv")
        written += [f"{prof.name}_train.csv", f"{prof.name}_demo.csv", f"{prof.name}_demo_truth.csv"]

        if prof.name == "average":
            t = np.arange(len(ecg)) / 250.0
            np.savetxt(f"{out}/ecg_average_demo.csv", np.c_[t, ecg], delimiter=",",
                       header="t_s,ecg_mv", comments="", fmt=["%.4f", "%.4f"])
            written.append("ecg_average_demo.csv")
            for lv in (0.10, 0.25, 0.50):
                bad = Corruptor(lv, seed=int(lv * 100)).df_corrupt(meas)
                bad.round(3).to_csv(f"{out}/average_demo_corrupt_{int(lv*100):02d}.csv", index=False)
                written.append(f"average_demo_corrupt_{int(lv*100):02d}.csv")

    # clean resting ECG for replay / R-peak testing
    rest = Simulator("average", Timeline("rest").add("rest", 60), seed=42)
    _, _, ecg = rest.collect(keep_ecg=True)
    np.savetxt(f"{out}/ecg_average_rest_60s.csv", np.c_[np.arange(len(ecg)) / 250.0, ecg],
               delimiter=",", header="t_s,ecg_mv", comments="", fmt=["%.4f", "%.4f"])
    written.append("ecg_average_rest_60s.csv")

    total = sum(os.path.getsize(f"{out}/{f}") for f in written)
    print(f"wrote {len(written)} files to {out}  ({total/1e6:.1f} MB)")
    return written


# ##########################################################################
# PART 6a - SELF-TESTS (was test_simulator.py)
# ##########################################################################

def test_get_frame_5_minutes():
    sim = Simulator("average", demo_5min())
    frames = [sim.get_frame() for _ in range(300)]
    assert all(k in frames[0] for k in SHARED_KEYS)
    assert all(len(f["ecg"]) == 250 for f in frames)                 # 1 s @ 250 Hz
    truth = sim.truth_df()
    hr_err = np.mean(np.abs(np.array([f["hr"] for f in frames]) - truth.hr.values))
    assert hr_err < 3.0, hr_err                                      # measured HR tracks true HR
    ecg = np.concatenate([f["ecg"] for f in frames])
    pk, _ = find_peaks(ecg, height=0.55, distance=int(0.25 * 250))
    n_beats = sum(len(f["beats"]) for f in frames)
    assert abs(len(pk) - n_beats) <= 3, (len(pk), n_beats)          # ECG waveform agrees with beat schedule
    walk, rest = truth[truth.label == "walking"].hr.mean(), truth[truth.label == "rest"].hr.mean()
    assert walk > rest + 20                                          # activity raises HR
    hyp = truth[truth.label == "hypoxia"]
    assert hyp.spo2.min() < 90 and (hyp.hr_residual_true.mean() > 10)  # hypoxia: SpO2 down, HR up vs normal
    print("ok  5-minute get_frame(): HR MAE %.2f bpm, %d beats" % (hr_err, n_beats))


def test_live_controls():
    sim = Simulator("average", demo_3min())
    for _ in range(10): sim.get_frame()
    sim.set_state(spo2=86, arrhythmia="pvc")
    for _ in range(40): f = sim.get_frame(with_truth=True)
    assert f["truth"]["anomaly"] == 1 and f["spo2"] < 92 and "+live" in f["truth"]["label"]
    sim.clear_overrides()
    assert not sim.overrides
    sim.set_lead_off(True); f = sim.get_frame()
    assert f["leads_ok"] is False and f["beats"] == []
    print("ok  live controls")


def test_corruption_levels():
    sim = Simulator("average", demo_5min()); meas, truth, ecg = sim.collect(keep_ecg=True)
    for lv in (0.0, 0.25, 0.5):
        c = Corruptor(lv, seed=1).ecg_corrupt(ecg)
        lost = 1 - c["valid"].mean(); off = 1 - c["leads_ok"].mean()
        assert abs(lost + off - lv) < 0.03, (lv, lost, off)
        assert np.array_equal(c["clean"], ecg)                       # ground truth untouched
    print("ok  corruption levels 0 / 0.25 / 0.5")


def test_profiles_differ():
    hr = [Simulator(n, demo_3min()).get_frame(with_ecg=False)["hr"] for n in ("athlete", "average", "anxious")]
    assert hr[0] < hr[1] < hr[2]
    print("ok  patient profiles differ (rest HR %s)" % [round(x) for x in hr])


# ##########################################################################
# PART 6 - TESTS + COMMAND LINE
# ##########################################################################

def run_tests():
    test_get_frame_5_minutes()
    test_live_controls()
    test_corruption_levels()
    test_profiles_differ()
    print("ALL PASSED")


def demo(profile="average", seconds=60):
    """Print a live-looking table of the 3-minute demo scenario."""
    sim = Simulator(profile, demo_3min())
    print(f"patient: {profile}   (script: rest 0-45s, walking 45-90s, recovery, hypoxia 105-150s)")
    print("  t    HR    HRV   SpO2  resp    BP   temp  scenario")
    for _ in range(int(seconds)):
        f = sim.get_frame(with_truth=True)
        print(f"{f['t']:3.0f} {f['hr']:5.1f} {f['hrv']:6.1f} {f['spo2']:6.1f} {f['resp']:5.1f} "
              f"{f['bp']:5.0f} {f['temp']:6.2f}  {f['truth']['label']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Person B: synthetic patient simulator (single file)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("test", help="run the self-test")
    d = sub.add_parser("demo", help="print the demo scenario")
    d.add_argument("--profile", default="average", choices=sorted(PROFILES))
    d.add_argument("--seconds", type=int, default=60)
    g = sub.add_parser("datasets", help="write the CSV datasets")
    g.add_argument("--out", default=OUT, help="output folder (default: <repo root>/data)")
    a = ap.parse_args(argv)
    if a.cmd == "test":
        run_tests()
    elif a.cmd == "demo":
        demo(a.profile, a.seconds)
    elif a.cmd == "datasets":
        generate_datasets(a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
