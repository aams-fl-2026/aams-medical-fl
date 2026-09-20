"""
Device Time Simulator
=====================
Implements Formula 4 from:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022

Formula 4 (paper exact):
  t_k = a_k + Exp(1/mu_k)

Where:
  a_k  ~ U(0.5, 2.0) — minimum training time in minutes
  mu_k ~ U(0.5, 2.0) — rate parameter (higher = less variance)
  Exp(1/mu_k)        — stochastic component in minutes

Round time = max_{k in V_m^r} {t_k}  (Formula 3)

This gives per-round times of 1-5 minutes naturally,
matching the paper's reported simulation times.
"""
import numpy as np


class DeviceTimeSimulator:
    def __init__(self, device_caps, device_num_samples, seed=42):
        """
        Args:
            device_caps: dict {device_id: {'capability': a_k, 'fluctuation': mu_k}}
            device_num_samples: not used in paper's Formula 4 but kept for compatibility
            seed: random seed
        """
        self.caps    = device_caps
        self.samples = device_num_samples
        self.rng     = np.random.default_rng(seed)

    def simulate_round(self, selected_devices, job_id, local_epochs):
        """
        Formula 4: t_k = a_k + Exp(1/mu_k)
        Formula 3: round_time = max(t_k)
        Returns time in minutes.
        """
        if not selected_devices:
            return 0.0

        device_times = []
        for d in selected_devices:
            a_k  = self.caps[d]['capability']   # minimum time (minutes)
            mu_k = self.caps[d]['fluctuation']  # rate parameter
            # Formula 4: t_k = a_k + Exp(1/mu_k)
            t_k = a_k + self.rng.exponential(1.0 / mu_k)
            device_times.append(t_k)

        # Formula 3: round time = slowest device
        return max(device_times)

    def get_device_times(self, selected_devices, job_id, local_epochs):
        """Return individual device times for DDR-RLDS reassignment logic."""
        times = {}
        for d in selected_devices:
            a_k  = self.caps[d]['capability']
            mu_k = self.caps[d]['fluctuation']
            times[d] = a_k + self.rng.exponential(1.0 / mu_k)
        return times