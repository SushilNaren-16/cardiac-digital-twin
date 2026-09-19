"""
twin/model.py
Person C's twin: synthetic patient data -> baseline predictor -> RLS
personalization -> residual-based alerts, compared against a fixed
HR>100 threshold.

This does NOT depend on B's simulator or A's firmware yet. It generates
its own synthetic data in the shared dict format from docs/protocol.md,
so it runs standalone. Once B's simulator or the live bridge is ready,
only generate_synthetic_data() gets swapped out -- everything downstream
stays the same.
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. SYNTHETIC DATA (stand-in for B's simulator / bridge output)
# ----------------------------------------------------------------------

def generate_synthetic_data(duration_s=180, dt=1.0, seed=42):
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    records = []

    baseline_hr = 68.0

    for i in range(n):
        t = i * dt

        if t < 60:
            activity = 0.1
            spo2 = 98.0
        elif t < 120:
            activity = 0.7
            spo2 = 97.5
        elif t < 150:
            activity = 0.2
            spo2 = 89.0
        else:
            activity = 0.15
            spo2 = 96.0

        resp = 12 + 8 * activity + rng.normal(0, 0.5)

        hr_true = (
            baseline_hr
            + 35 * activity
            + 0.3 * (resp - 12)
            + max(0, (95 - spo2)) * 1.8
            + rng.normal(0, 1.5)
        )

        records.append({
            "t": t,
            "hr": hr_true,
            "resp": resp,
            "spo2": spo2,
            "activity": activity,
            "source": "synthetic",
        })

    return records


# ----------------------------------------------------------------------
# 2. TWIN v0 -- predicts expected HR from activity, resp, spo2
# ----------------------------------------------------------------------

class RLSTwin:
    def __init__(self, n_features=4, forgetting_factor=0.99):
        self.w = np.array([70.0, 20.0, 0.2, 1.0])
        self.P = np.eye(n_features) * 10.0
        self.lam = forgetting_factor
        self.n_seen = 0

    def _features(self, activity, resp, spo2):
        return np.array([1.0, activity, resp, max(0, 95 - spo2)])

    def predict(self, activity, resp, spo2):
        x = self._features(activity, resp, spo2)
        return float(self.w @ x)

    def update(self, activity, resp, spo2, hr_measured, freeze=False):
        self.n_seen += 1
        if freeze:
            return

        x = self._features(activity, resp, spo2)
        Px = self.P @ x
        gain = Px / (self.lam + x @ Px)
        error = hr_measured - float(self.w @ x)
        self.w = self.w + gain * error
        self.P = (self.P - np.outer(gain, Px)) / self.lam


# ----------------------------------------------------------------------
# 3. RESIDUAL ALERTING -- frozen calm baseline, not a rolling window
# ----------------------------------------------------------------------

class ResidualAlerter:
    def __init__(self, calm_window=45, threshold_sigmas=4.0):
        self.calm_window = calm_window
        self.threshold_sigmas = threshold_sigmas
        self.calibration = []
        self.baseline_mu = None
        self.baseline_sigma = None

    def check(self, predicted_hr, measured_hr):
        residual = measured_hr - predicted_hr

        if self.baseline_mu is None:
            self.calibration.append(residual)
            if len(self.calibration) >= self.calm_window:
                self.baseline_mu = np.mean(self.calibration)
                self.baseline_sigma = np.std(self.calibration) + 1e-6
            return False, residual

        z = abs(residual - self.baseline_mu) / self.baseline_sigma
        is_alert = z > self.threshold_sigmas
        return is_alert, residual


# ----------------------------------------------------------------------
# 4. RUN THE PIPELINE + COMPARE AGAINST FIXED THRESHOLD
# ----------------------------------------------------------------------

def run():
    data = generate_synthetic_data()

    twin = RLSTwin()
    alerter = ResidualAlerter(calm_window=45, threshold_sigmas=3)

    times, hr_true, hr_pred, alerts_twin, alerts_fixed = [], [], [], [], []

    for rec in data:
        pred = twin.predict(rec["activity"], rec["resp"], rec["spo2"])
        is_alert, residual = alerter.check(pred, rec["hr"])

        twin.update(rec["activity"], rec["resp"], rec["spo2"], rec["hr"],
                    freeze=False)

        fixed_alert = rec["hr"] > 100

        times.append(rec["t"])
        hr_true.append(rec["hr"])
        hr_pred.append(pred)
        alerts_twin.append(is_alert)
        alerts_fixed.append(fixed_alert)

    times = np.array(times)
    hr_true = np.array(hr_true)
    hr_pred = np.array(hr_pred)
    alerts_twin = np.array(alerts_twin)
    alerts_fixed = np.array(alerts_fixed)

    walking = (times >= 60) & (times < 120)
    hypoxia = (times >= 120) & (times < 150)

    print("=== RESULTS ===")
    print(f"False alarms during walking (twin, residual-based): {alerts_twin[walking].sum()}")
    print(f"False alarms during walking (fixed HR>100 threshold): {alerts_fixed[walking].sum()}")
    print(f"Alerts caught during hypoxia (twin): {alerts_twin[hypoxia].sum()} / {hypoxia.sum()} samples")
    print(f"Alerts caught during hypoxia (fixed threshold): {alerts_fixed[hypoxia].sum()} / {hypoxia.sum()} samples")
    print(f"Learned baseline weights (w0..w3): {np.round(twin.w, 2)}")

    plt.figure(figsize=(10, 5))
    plt.plot(times, hr_true, label="Measured HR", color="black")
    plt.plot(times, hr_pred, label="Twin predicted HR", color="tab:blue", linestyle="--")
    plt.scatter(times[alerts_twin], hr_true[alerts_twin], color="red", zorder=5, label="Twin alert")
    plt.axvspan(60, 120, color="green", alpha=0.1, label="Walking (should NOT alert)")
    plt.axvspan(120, 150, color="orange", alpha=0.15, label="Hypoxia (SHOULD alert)")
    plt.xlabel("Time (s)")
    plt.ylabel("HR (bpm)")
    plt.title("Personalized Twin: Measured vs Predicted HR, with Residual Alerts")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("twin_results.png")
    print("\nSaved plot to twin_results.png")


if __name__ == "__main__":
    run()