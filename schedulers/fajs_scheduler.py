#!/usr/bin/env python
"""
Multi-Job Federated Learning — FAJS Scheduler (Shi & Yu, ICASSP 2024)
========================================================================
Implements the device + job scheduling policy from:
  Y. Shi and H. Yu, "Fairness-Aware Job Scheduling for Multi-Job
  Federated Learning," IEEE ICASSP 2024, pp. 6350-6354.

This paper extends Zhou et al.'s multi-job FL framework (AAAI-22) — the
same base framework RLDS/BODS/FedCS/AAMS build on — by recognizing that
clients can be shared across MULTIPLE job types simultaneously rather
than being exclusively assigned to one job per round, and that fairness
should be measured and balanced at the JOB level (not just the device
level as in RLDS's Formula 5).

Key differences from RLDS:
  1. Job-level fairness term J_fair(r): measures how evenly the *jobs*
     have received aggregation opportunities (rounds-completed ratio),
     not just how evenly devices have been picked.
  2. A "job urgency" weight u_m(r) is computed from how far behind a
     job's relative progress is versus the average across all jobs —
     jobs that are falling behind get prioritized for the *next*
     device-allocation pass, even if their devices were already used
     this round elsewhere.
  3. Device-job assignment is solved via a greedy auction: jobs bid
     for devices in order of urgency u_m(r), and devices go to the job
     with the highest (urgency x suitability) bid, instead of RLDS's
     single shared LSTM policy. This keeps the comparison "synchronous,
     greedy, fairness-driven" instead of RL-driven (matching the paper's
     actual non-RL scheduling formulation).

Cost / scheduling model:
    bid_{m,k}(r) = u_m(r) * (1 / t_k)               # urgency x speed
    u_m(r)       = 1 + lambda * (avg_progress - progress_m(r))
    progress_m(r)= acc_m(r) / target_m              # normalized progress
  Device k assigned to job m* = argmax_m bid_{m,k}(r), subject to each
  job receiving at most K devices/round and each device used by at most
  one job per round (Hungarian-style greedy auction, paper Section III).

  Round time (Formula 4, same shift-exponential as RLDS/BODS):
      t_k = a_k + Exp(1/mu_k)

This scheduler is SYNCHRONOUS: each job still waits for ALL of its
assigned devices to finish before aggregating (no async K_min, no
straggler reuse) — same limitation AAMS addresses.

Jobs (Group B, 3-job setting):
  Job 0 : ResNet-18  + CIFAR-10           target = 54.6%
  Job 1 : CNN-B      + Fashion-MNIST      target = 82.1%
  Job 2 : AlexNet    + EMNIST-Balanced    target = 70.0%
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet             import ResNet18
from models.cnn_b              import CNNB
from models.alexnet            import AlexNet
from models.vgg               import VGG11
from federated.client          import FLClient
from federated.server          import FLServer
from models.non_iid_partition  import create_non_iid_datasets
from utils.time_simulator      import DeviceTimeSimulator


NUM_JOBS = 4

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ── Global plot state ─────────────────────────────────────────────────────────
_fig = _axes = None
_time_history = {j: [] for j in range(NUM_JOBS)}
_acc_history  = {j: [] for j in range(NUM_JOBS)}
_loss_history = {j: [] for j in range(NUM_JOBS)}
_JOB_COLORS   = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple'}
_JOB_NAMES    = {
    0: 'ResNet18/CIFAR-10',
    1: 'CNN-B/FashionMNIST',
    2: 'AlexNet/EMNIST-Balanced',
    3: 'AlexNet/CIFAR-100',
    4: 'VGG-11/EMNIST-Letters',
}

def setup_plots():
    global _fig, _axes
    if not PLOT_AVAILABLE:
        return
    plt.ion()
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig.suptitle('FAJS Scheduler (Shi & Yu, ICASSP 2024) — Training Progress', fontsize=13)
    for ax, title, ylabel in [
        (_axes[0], 'Test Accuracy vs Simulated Time', 'Accuracy (%)'),
        (_axes[1], 'Test Loss vs Simulated Time',     'Loss'),
    ]:
        ax.set_xlabel('Simulated Time (min)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def update_plots(job_id, sim_time_min, acc, loss):
    _time_history[job_id].append(sim_time_min)
    _acc_history[job_id].append(acc)
    _loss_history[job_id].append(loss)
    if not PLOT_AVAILABLE or _axes is None:
        return
    _axes[0].cla(); _axes[1].cla()
    for j in range(NUM_JOBS):
        if _time_history[j]:
            _axes[0].plot(_time_history[j], _acc_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
            _axes[1].plot(_time_history[j], _loss_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
    for ax, ylabel in [(_axes[0], 'Accuracy (%)'), (_axes[1], 'Loss')]:
        ax.set_xlabel('Simulated Time (min)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _axes[0].set_title('Test Accuracy vs Simulated Time')
    _axes[1].set_title('Test Loss vs Simulated Time')
    _fig.canvas.draw()
    _fig.canvas.flush_events()


def save_plots():
    if not PLOT_AVAILABLE or _fig is None:
        return
    plt.ioff()
    os.makedirs('results', exist_ok=True)
    path = 'results/fajs_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {path}')


# ── FAJS Scheduler (Shi & Yu, ICASSP 2024) ───────────────────────────────────
class FAJSScheduler:
    """
    Fairness-Aware Job Scheduling for multi-job FL (Shi & Yu, ICASSP 2024).

    Differs from RLDS/BODS in two ways:
      1. No learned policy — devices are assigned via a greedy auction
         each round, biased by per-job "urgency" (how far behind a job
         is relative to the average progress across all active jobs).
      2. Fairness is enforced at the JOB level: u_m(r) grows the longer
         a job lags behind, ensuring no job is starved of devices over
         many rounds (paper's core contribution vs. RLDS, which only
         balances *device* selection frequency, not *job* progress).

    Cost model (kept identical to RLDS/BODS/AAMS for fair comparison):
        t_k = a_k + Exp(1/mu_k)                      [Formula 4]
        T_r = max_{k in V_m} t_k                      (synchronous wait)
    """

    def __init__(self, num_devices, devices_per_round, num_jobs,
                 device_caps, job_targets, simulator=None, lam=2.0, seed=42):
        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.num_jobs          = num_jobs
        self.device_caps       = device_caps      # {d: {'capability': a_k, 'fluctuation': mu_k}}
        self.job_targets       = job_targets       # {j: target_acc}
        self.simulator          = simulator        # DeviceTimeSimulator (Formula 4)
        self.lam                = lam              # urgency sensitivity (paper's lambda)
        self.rng                = np.random.default_rng(seed)

        # Track progress (normalized accuracy) and selection counts per job
        self.progress         = np.zeros(num_jobs)
        self.selection_counts = np.zeros((num_devices, num_jobs))
        self.rounds_completed = np.zeros(num_jobs)

    # ── Job urgency: how far behind average progress (paper Eq. 3) ──────────
    def _urgency(self):
        active_mask = self.progress < 1.0
        if not active_mask.any():
            return np.ones(self.num_jobs)
        avg_progress = self.progress[active_mask].mean()
        urgency = 1.0 + self.lam * np.clip(avg_progress - self.progress, -1, 1)
        urgency = np.where(active_mask, urgency, 0.0)   # finished jobs bid 0
        return urgency

    # ── Greedy auction: devices bid out to highest (urgency x speed) job ────
    def select_devices(self, occupied, job_done):
        """
        Greedy auction (paper Section III-B):
          For each device k not yet occupied, compute its bid value for
          every still-active job m: bid_{m,k} = u_m * (1 / E[t_k]) * decay(k,m).
          Assign devices one at a time to the (job, device) pair with the
          highest remaining bid, until every active job has K devices or
          no devices remain. This approximates the paper's Hungarian-style
          assignment with a fast greedy heuristic (equivalent solution
          quality for the K << N regime used here).

        A selection-count decay term is included so that a device which has
        already been picked many times for a given job becomes progressively
        less attractive relative to less-used devices. Without this term the
        bid value bid_{m,k} = urgency_m * speed_k depends only on (m, k) and
        is therefore IDENTICAL every round for as long as urgency_m doesn't
        change — causing the same top-K fastest devices to win every single
        round indefinitely (observed empirically as device-selection sigma
        growing unboundedly with round count, with Job 0/Job 2 plateauing
        because they only ever see a handful of devices' non-IID shards).
        The paper's own fairness objective (job-level progress balance) is
        preserved; this only prevents *device-level* starvation, matching
        the spirit of RLDS's Formula 5 fairness term that FAJS's design
        otherwise omits at the device level.
        """
        urgency = self._urgency()
        available = [d for d in range(self.num_devices) if d not in occupied]

        # Precompute expected speed (1 / mean completion time) per device
        speed = {}
        for d in available:
            a, mu = self.device_caps[d]['capability'], self.device_caps[d]['fluctuation']
            expected_t = a + (1.0 / mu if mu > 0 else 1.0)
            speed[d] = 1.0 / max(expected_t, 1e-6)

        # Build all (job, device) bids for active jobs, penalized by how many
        # times this (job, device) pair has already been selected so the
        # auction is forced to rotate through the device pool over time.
        bids = []
        for m in range(self.num_jobs):
            if job_done[m] or urgency[m] <= 0:
                continue
            for d in available:
                decay = 1.0 / (1.0 + self.selection_counts[d, m])
                # small random jitter breaks any remaining exact ties so the
                # same device doesn't always win when bids are equal
                jitter = 1.0 + 1e-3 * self.rng.standard_normal()
                bids.append((urgency[m] * speed[d] * decay * jitter, m, d))
        bids.sort(reverse=True)   # highest bid first

        selections   = {j: [] for j in range(self.num_jobs)}
        used_devices = set()
        job_full     = {j: False for j in range(self.num_jobs)}

        for bid_val, m, d in bids:
            if job_full[m] or d in used_devices:
                continue
            selections[m].append(d)
            used_devices.add(d)
            if len(selections[m]) >= self.devices_per_round:
                job_full[m] = True

        return selections

    # ── Simulate round (max) time via the shared DeviceTimeSimulator ────────
    def round_time(self, selected, job_id=0):
        """Formula 4: t_k = a_k + Exp(1/mu_k); round time = max(t_k)."""
        if not selected:
            return 0.0
        if self.simulator is not None:
            return self.simulator.simulate_round(selected, job_id, local_epochs=None)
        # Fallback (only used if no simulator was supplied)
        times = []
        for d in selected:
            a, mu = self.device_caps[d]['capability'], self.device_caps[d]['fluctuation']
            t = a + self.rng.exponential(1.0 / mu if mu > 0 else 1.0)
            times.append(t)
        return max(times)

    # ── Update bookkeeping after a round ─────────────────────────────────────
    def update(self, selections, accuracies):
        for j, selected in selections.items():
            for d in selected:
                self.selection_counts[d, j] += 1
            if selected:
                self.rounds_completed[j] += 1
        for j, acc in accuracies.items():
            target = self.job_targets[j]
            self.progress[j] = min(acc / target, 1.0) if target > 0 else 0.0

    def get_fairness_stats(self):
        stats = {}
        for j in range(self.num_jobs):
            freq = self.selection_counts[:, j]
            stats[j] = {'std': float(np.std(freq)), 'min': int(freq.min()), 'max': int(freq.max())}
        return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    NUM_DEVICES        = 200
    DEVICES_PER_ROUND  = 10
    LOCAL_EPOCHS       = 5
    MAX_ROUNDS         = 5000
    SEED               = 42
    LAMBDA_URGENCY     = 2.0   # job-urgency sensitivity (paper's lambda)

    BATCH_SIZE       = {0: 30,  1: 10,  2: 64,  3: 64,  4: 64}
    LEARNING_RATE_FL = {0: 0.1, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}

    JOBS = {
        0: ('resnet18', 'cifar10',          3, 32, 60.0,  8),
        1: ('cnn_b',    'fashion_mnist',    1, 28, 85.0,  8),
        2: ('alexnet',  'emnist_balanced',  1, 28, 70.0, 38),
        3: ('alexnet',  'cifar100',         3, 32, 40.0, 80),
        4: ('vgg11',    'emnist_letters',   1, 28, 85.0,  2),
    }
    JOB_NAMES = {
        0: 'ResNet18 + CIFAR-10',
        1: 'CNN-B + FashionMNIST',
        2: 'AlexNet + EMNIST-Balanced',
        3: 'AlexNet + CIFAR-100',
        4: 'VGG-11 + EMNIST-Letters',
    }

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('FAJS SCHEDULER (Shi & Yu, ICASSP 2024) — Multi-Job Federated Learning')
    print('=' * 70)
    print(f'Device: {device}   NUM_JOBS: {NUM_JOBS}')
    for j in range(NUM_JOBS):
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  target={JOBS[j][4]}%')

    # ── Non-IID datasets ──────────────────────────────────────────────────────
    print('\nCreating non-IID datasets (2 classes/device)...')
    job_client_datasets, job_test_datasets = {}, {}
    for j in range(NUM_JOBS):
        _, dataset, _, _, _, classes_per_device = JOBS[j]
        client_data, test_data = create_non_iid_datasets(
            dataset, NUM_DEVICES, num_classes_per_device=classes_per_device, seed=SEED + j
        )
        job_client_datasets[j] = client_data
        job_test_datasets[j]   = test_data
        print(f'  Job {j} ({dataset}): {len(test_data)} test samples')

    # ── Models & servers ──────────────────────────────────────────────────────
    print('\nInitialising models...')
    servers = {}
    for j in range(NUM_JOBS):
        model_key, dataset, ch, sz, _, _ = JOBS[j]
        num_classes = 47 if dataset == 'emnist_balanced' else (100 if dataset == 'cifar100' else 10)
        if model_key == 'resnet18':
            model = ResNet18(num_classes=num_classes, input_channels=ch)
        elif model_key == 'cnn_b':
            model = CNNB(num_classes=num_classes)
        elif model_key == 'alexnet':
            model = AlexNet(num_classes=num_classes, input_channels=ch, input_size=sz)
        elif model_key == 'vgg11':
            model = VGG11(num_classes=26, input_channels=ch)
        else:
            raise ValueError(f'Unknown model: {model_key}')
        servers[j] = FLServer(model.to(device), job_test_datasets[j], device=str(device))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  params={n_params:,}')

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    for j in range(NUM_JOBS):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d],
                        BATCH_SIZE[j], LEARNING_RATE_FL[j], device)
            for d in range(NUM_DEVICES)
        }

    # ── Mixed device heterogeneity (matches Oort/PoC/AAMS exactly) ───────────
    rng      = np.random.default_rng(SEED)
    n_slow   = int(0.3 * NUM_DEVICES)
    n_medium = int(0.4 * NUM_DEVICES)
    n_fast   = NUM_DEVICES - n_slow - n_medium
    caps   = np.concatenate([rng.uniform(2.0, 5.0, n_slow),
                              rng.uniform(0.5, 2.0, n_medium),
                              rng.uniform(0.1, 0.5, n_fast)])
    flucts = np.concatenate([rng.uniform(0.1, 0.5, n_slow),
                              rng.uniform(0.5, 1.5, n_medium),
                              rng.uniform(2.0, 5.0, n_fast)])
    rng.shuffle(caps); rng.shuffle(flucts)
    device_caps = {
        d: {'capability': float(caps[d]), 'fluctuation': float(flucts[d])}
        for d in range(NUM_DEVICES)
    }

    device_num_samples = {
        j: {d: len(job_client_datasets[j][d]) for d in range(NUM_DEVICES)}
        for j in range(NUM_JOBS)
    }
    simulator = DeviceTimeSimulator(device_caps, device_num_samples, seed=SEED)

    job_targets = {j: JOBS[j][4] for j in range(NUM_JOBS)}

    # ── FAJS Scheduler ────────────────────────────────────────────────────────
    print('\nInitialising FAJS scheduler...')
    scheduler = FAJSScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=NUM_JOBS,
        device_caps=device_caps,
        job_targets=job_targets,
        simulator=simulator,
        lam=LAMBDA_URGENCY,
        seed=SEED,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)
    setup_plots()

    log = {j: {'sim_time': [], 'acc': [], 'loss': []} for j in range(NUM_JOBS)}
    job_done      = {j: False for j in range(NUM_JOBS)}
    job_sim_time  = {j: 0.0   for j in range(NUM_JOBS)}
    job_rounds    = {j: 0     for j in range(NUM_JOBS)}
    job_final_acc = {j: 0.0   for j in range(NUM_JOBS)}
    round_num     = 0
    global_sim_time = 0.0

    with tqdm(total=MAX_ROUNDS, desc='FAJS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1

            # ── Greedy fairness-aware auction (paper Section III) ────────────
            selections = scheduler.select_devices(occupied=set(), job_done=job_done)

            round_accs = {}
            round_time_this_round = 0.0

            for j in range(NUM_JOBS):
                if job_done[j]:
                    round_accs[j] = JOBS[j][4]
                    continue

                selected = selections.get(j, [])
                if not selected:
                    continue

                # Synchronous round time = max device completion time
                t_round = scheduler.round_time(selected, job_id=j)
                round_time_this_round = max(round_time_this_round, t_round)

                # Local training + FedAvg (all selected devices wait for slowest)
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in selected
                ]
                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                result = servers[j].evaluate()
                acc, loss = result['test_accuracy'], result['test_loss']
                job_final_acc[j] = acc
                round_accs[j] = acc

                job_sim_time[j] += t_round
                update_plots(j, job_sim_time[j], acc, loss)
                log[j]['sim_time'].append(job_sim_time[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                target = JOBS[j][4]
                if acc >= target and not job_done[j]:
                    job_done[j] = True
                    print(f'\n  Job {j} ({JOB_NAMES[j]}) reached {target}%'
                          f' at round {job_rounds[j]}'
                          f' ({job_sim_time[j]:.1f} min)')

            global_sim_time += round_time_this_round
            scheduler.update(selections, round_accs)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(NUM_JOBS) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    # ── Save log ──────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    log_path = 'results/fajs_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f)
    print(f'  Log saved -> {log_path}')

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = sum(job_sim_time.values())
    print('\n' + '=' * 70)
    print('FAJS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<28} {"Time (min)":>12} '
          f'{"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(NUM_JOBS):
        print(f'  {j}    {JOB_NAMES[j]:<28} '
              f'{job_sim_time[j]:>10.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
    print('-' * 70)
    print(f'  Total time: {total_time:.1f} min  ({total_time/60:.2f} h)')

    stats = scheduler.get_fairness_stats()
    print('\nFairness (sigma of per-device selection counts):')
    for j in range(NUM_JOBS):
        print(f'  Job {j}: sigma={stats[j]["std"]:.2f}  '
              f'min={stats[j]["min"]}  max={stats[j]["max"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()