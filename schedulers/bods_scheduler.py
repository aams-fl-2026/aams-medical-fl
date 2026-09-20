#!/usr/bin/env python
"""
Multi-Job Federated Learning — BODS Scheduler
==============================================
Implements Algorithm 1 from:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

BODS (Bayesian Optimization-based Device Scheduling):
  - Gaussian Process with Matern kernel fits TotalCost function
  - Expected Improvement (EI) acquisition function (Formula 14-15)
  - Randomly samples candidate scheduling plans each round
  - Selects plan with highest EI (most improvement over best so far)
  - One BODS instance per job (paper Figure 1)
  - Same cost model as RLDS: Formula 2 (alpha*time + beta*fairness)

Uses Formula 4 shift-exponential time simulation.
Fixed rounds for fair comparison:
  Job 0 (ResNet/CIFAR-10)    : 200 rounds
  Job 1 (CNN-B/FashionMNIST) : 500 rounds
  Job 2 (AlexNet/MNIST)      : 900 rounds

Paper settings (Table 4):
  - 100 devices, 10 per round per job
  - 5 local epochs
  - Batch sizes: ResNet=30, CNN-B=10, AlexNet=64
  - Learning rates: ResNet=0.1, CNN-B=0.01, AlexNet=0.01
  - FedAvg aggregation
  - Non-IID: 2 classes per device
  - alpha=0.5, beta=0.5 (Formula 2)
"""
import os
import sys
import random
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet            import ResNet18
from models.cnn_b             import CNNB
from models.alexnet           import AlexNet
from models.vgg               import VGG11
from federated.client         import FLClient
from federated.server         import FLServer
from models.non_iid_partition import create_non_iid_datasets
from utils.time_simulator     import DeviceTimeSimulator

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm


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
_time_history = {j: [] for j in range(4)}
_acc_history  = {j: [] for j in range(4)}
_loss_history = {j: [] for j in range(4)}
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
    _fig.suptitle('BODS Scheduler — Training Progress', fontsize=13)
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


def update_plots(job_id, sim_time, acc, loss):
    _time_history[job_id].append(sim_time)
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
        ax.set_xlabel('Simulated Time (min)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    _axes[0].set_title('Test Accuracy vs Simulated Time')
    _axes[1].set_title('Test Loss vs Simulated Time')
    _fig.canvas.draw()
    _fig.canvas.flush_events()


def save_plots():
    if _fig is None:
        return
    os.makedirs('results', exist_ok=True)
    path = 'results/bods_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {path}')


# ── BODS Agent (Algorithm 1) ──────────────────────────────────────────────────
class BODSAgent:
    """
    One BODS agent per job.

    Algorithm 1:
    1. Randomly generate initial observation points (scheduling plans + costs)
    2. For each round:
       a. Sample N candidate plans from available devices
       b. Select plan with max EI using updated GP
       c. Perform FL training with selected plan
       d. Calculate real cost, add to observation set
    """
    def __init__(self, job_id, num_devices, devices_per_round,
                 device_caps, alpha=0.5, beta=0.5,
                 n_initial=10, n_candidates=20, seed=42):
        self.job_id            = job_id
        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.device_caps       = device_caps
        self.alpha             = alpha
        self.beta              = beta
        self.n_initial         = n_initial    # initial observation points
        self.n_candidates      = n_candidates # candidate plans per round
        self.rng               = np.random.default_rng(seed + job_id)

        # Selection frequency for fairness tracking
        self.selection_freq = np.zeros(num_devices)

        # GP with Matern kernel (paper Section 4)
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

        # Observation history: X = plan features, y = costs
        self.obs_X = []   # list of feature vectors
        self.obs_y = []   # list of cost values
        self.best_cost = np.inf

        # Build capability arrays
        self.cap_array   = np.array([device_caps[d]['capability']
                                     for d in range(num_devices)])
        self.fluct_array = np.array([device_caps[d]['fluctuation']
                                     for d in range(num_devices)])

    def _plan_to_features(self, plan):
        """
        Convert a scheduling plan to a feature vector for GP.
        Features: mean capability, mean fluctuation, fairness cost,
                  fraction of high-cap devices selected.
        """
        caps   = self.cap_array[plan]
        flucts = self.fluct_array[plan]

        # Temporary freq to compute fairness if this plan is selected
        tmp_freq = self.selection_freq.copy()
        for d in plan:
            tmp_freq[d] += 1
        total = tmp_freq.sum()
        if total > 0:
            norm_freq = tmp_freq / (total + 1e-8)
            fairness  = float(np.mean((norm_freq - np.mean(norm_freq)) ** 2))
        else:
            fairness = 0.0

        # High capability fraction
        mean_cap = float(np.mean(self.cap_array))
        high_cap_frac = float(np.mean(caps > mean_cap))

        return np.array([
            float(np.mean(caps)),
            float(np.std(caps)),
            float(np.mean(flucts)),
            fairness,
            high_cap_frac,
        ])

    def _time_cost(self, plan):
        """Formula 3: max device execution time."""
        times = []
        for d in plan:
            a  = self.device_caps[d]['capability']
            mu = self.device_caps[d]['fluctuation']
            t  = a + np.random.exponential(1.0 / mu if mu > 0 else 1.0)
            times.append(t)
        return max(times) if times else 0.0

    def _fairness_cost(self):
        """Formula 5: normalized variance of selection frequency."""
        total = self.selection_freq.sum()
        if total == 0:
            return 0.0
        normalized = self.selection_freq / (total + 1e-8)
        return float(np.mean((normalized - np.mean(normalized)) ** 2))

    def _total_cost(self, plan):
        """Formula 2: alpha * time_cost + beta * fairness_cost."""
        return self.alpha * self._time_cost(plan) + self.beta * self._fairness_cost()

    def _expected_improvement(self, X_candidates):
        """
        Formula 14-15: EI acquisition function.
        EI(V) = E[max(0, C+_{L-1} - TotalCost(V))]
        """
        if len(self.obs_X) < 2:
            # Not enough observations for GP — return uniform
            return np.ones(len(X_candidates))

        X_obs = np.array(self.obs_X)
        y_obs = np.array(self.obs_y)

        try:
            self.gp.fit(X_obs, y_obs)
            mu, sigma = self.gp.predict(X_candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-8)

            # Best observed cost so far
            best = self.best_cost

            # EI formula (we minimize cost, so improvement = best - mu)
            improvement = best - mu
            Z  = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei = np.maximum(ei, 0.0)
            return ei
        except Exception:
            return np.ones(len(X_candidates))

    def initialize(self, occupied):
        """
        Algorithm 1 Line 1: randomly generate initial observation points.
        Returns the best initial plan.
        """
        available = [d for d in range(self.num_devices) if d not in occupied]
        n         = min(self.devices_per_round, len(available))

        best_plan = None
        best_cost = np.inf

        for _ in range(self.n_initial):
            plan = self.rng.choice(available, size=n, replace=False).tolist()
            cost = self._total_cost(plan)
            feat = self._plan_to_features(plan)
            self.obs_X.append(feat)
            self.obs_y.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_plan = plan

        self.best_cost = min(self.obs_y)
        return best_plan if best_plan else self.rng.choice(
            available, size=n, replace=False).tolist()

    def select(self, occupied):
        """
        Algorithm 1 Lines 3-4: sample candidate plans, pick max EI.
        """
        available = [d for d in range(self.num_devices) if d not in occupied]
        if not available:
            return []

        n = min(self.devices_per_round, len(available))

        # Sample n_candidates random plans (Algorithm 1 Line 3)
        candidates = []
        for _ in range(self.n_candidates):
            plan = self.rng.choice(available, size=n, replace=False).tolist()
            candidates.append(plan)

        # Compute features for all candidates
        X_candidates = np.array([self._plan_to_features(p) for p in candidates])

        # Select plan with max EI (Algorithm 1 Line 4)
        ei_values = self._expected_improvement(X_candidates)
        best_idx  = np.argmax(ei_values)
        selected  = candidates[best_idx]

        return selected

    def update(self, selected):
        """
        Algorithm 1 Lines 5-7: perform FL, compute real cost, update GP.
        """
        # Compute real cost after selection
        real_cost = self._total_cost(selected)
        feat      = self._plan_to_features(selected)

        # Update observation set (Algorithm 1 Line 7)
        self.obs_X.append(feat)
        self.obs_y.append(real_cost)
        self.best_cost = min(self.obs_y)

        # Update selection frequency
        for d in selected:
            self.selection_freq[d] += 1

    def get_fairness_stats(self):
        counts = self.selection_freq
        return {
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts)),
        }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('MULTI-JOB BODS SCHEDULER  (paper: Zhou et al. AAAI-22)')
    print('=' * 70)

    SEED  = 42
    ALPHA = 0.5
    BETA  = 0.5

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice : {device}')

    # ── Paper settings (Table 4) ──────────────────────────────────────────────
    NUM_DEVICES        = 100
    DEVICES_PER_ROUND  = 10
    LOCAL_EPOCHS       = 5
    BATCH_SIZE       = {0: 30,  1: 10,  2: 64,  3: 64,  4: 64}
    LEARNING_RATE      = {0: 0.1, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}
    MAX_ROUNDS_PER_JOB = {0: 200, 1: 500, 2: 900, 3: 1200, 4: 1500}

    # BODS specific
    N_INITIAL    = 10   # initial random observations
    N_CANDIDATES = 20   # candidate plans per round

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

    print('\nJobs:')
    for j, (m, d, _, _, t, _) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  max_rounds={MAX_ROUNDS_PER_JOB[j]}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    print('\nCreating non-IID datasets (2 classes/device)...')
    job_client_datasets = {}
    job_test_datasets   = {}
    for j, (_, dataset, _, _, _, nc) in JOBS.items():
        client_data, test_data = create_non_iid_datasets(
            dataset, NUM_DEVICES, num_classes_per_device=nc, seed=SEED + j
        )
        job_client_datasets[j] = client_data
        job_test_datasets[j]   = test_data
        print(f'  Job {j} ({dataset}): {len(test_data)} test samples')

    # ── Models & servers ──────────────────────────────────────────────────────
    print('\nInitialising models...')
    servers = {}
    for j, (model_key, _, ch, sz, _, _) in JOBS.items():
        if model_key == 'resnet18':
            dataset_name = JOBS[j][1]
            num_cls = 100 if dataset_name == 'cifar100' else 10
            model = ResNet18(num_classes=num_cls, input_channels=ch)
        elif model_key == 'cnn_b':
            model = CNNB(num_classes=10)
        elif model_key == 'alexnet':
            dataset_name = JOBS[j][1]
            num_cls = 100 if dataset_name == 'cifar100' else (47 if dataset_name == 'emnist_balanced' else 10)
            model = AlexNet(num_classes=num_cls, input_channels=ch, input_size=sz)
        elif model_key == 'vgg11':
            model = VGG11(num_classes=26, input_channels=ch)
        else:
            raise ValueError(f'Unknown model: {model_key}')
        servers[j] = FLServer(model.to(device), job_test_datasets[j], device=str(device))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  params={n_params:,}')

    # ── Device capabilities — Formula 4 ──────────────────────────────────────
    rng = np.random.default_rng(SEED)
    device_caps = {
        d: {
            'capability':  float(rng.uniform(0.5, 2.0)),
            'fluctuation': float(rng.uniform(0.1, 1.0)),
        }
        for d in range(NUM_DEVICES)
    }

    device_num_samples = {}
    for j in range(NUM_JOBS):
        device_num_samples[j] = {
            d: len(job_client_datasets[j][d])
            for d in range(NUM_DEVICES)
        }

    simulator = DeviceTimeSimulator(device_caps, device_num_samples, seed=SEED)

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    for j in range(NUM_JOBS):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d],
                        BATCH_SIZE[j], LEARNING_RATE[j], device)
            for d in range(NUM_DEVICES)
        }
        print(f'  Job {j}: {NUM_DEVICES} clients created')

    # One BODS agent per job
    print('\nInitialising BODS agents (one per job)...')
    agents = {
        j: BODSAgent(
            job_id=j,
            num_devices=NUM_DEVICES,
            devices_per_round=DEVICES_PER_ROUND,
            device_caps=device_caps,
            alpha=ALPHA,
            beta=BETA,
            n_initial=N_INITIAL,
            n_candidates=N_CANDIDATES,
            seed=SEED,
        )
        for j in range(NUM_JOBS)
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()

    log           = {j: {'sim_time': [], 'acc': [], 'loss': []} for j in range(NUM_JOBS)}
    job_done      = {j: False for j in range(NUM_JOBS)}
    job_sim_time  = {j: 0.0 for j in range(NUM_JOBS)}
    job_times     = {j: 0.0 for j in range(NUM_JOBS)}
    job_rounds    = {j: 0   for j in range(NUM_JOBS)}
    job_final_acc = {j: 0.0 for j in range(NUM_JOBS)}
    round_num     = 0
    max_rounds    = max(MAX_ROUNDS_PER_JOB.values())
    initialized   = {j: False for j in range(NUM_JOBS)}

    with tqdm(total=max_rounds, desc='BODS') as pbar:
        while not all(job_done.values()) and round_num < max_rounds:
            round_num += 1
            occupied = set()

            for j in range(NUM_JOBS):
                if job_done[j]:
                    continue

                # Algorithm 1: initialize with random observations first round
                if not initialized[j]:
                    selected = agents[j].initialize(occupied)
                    initialized[j] = True
                else:
                    # Algorithm 1 Lines 3-4: select via EI
                    selected = agents[j].select(occupied)

                if not selected:
                    continue

                occupied.update(selected)

                # Formula 4: simulate round time
                round_time = simulator.simulate_round(selected, j, LOCAL_EPOCHS)
                job_sim_time[j] += round_time

                # Local training + FedAvg
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in selected
                ]
                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                result = servers[j].evaluate()
                acc  = result['test_accuracy']
                loss = result['test_loss']
                job_final_acc[j] = acc

                update_plots(j, job_sim_time[j], acc, loss)

                # Algorithm 1 Lines 5-7: compute real cost, update GP
                agents[j].update(selected)

                log[j]['sim_time'].append(job_sim_time[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                if job_rounds[j] >= MAX_ROUNDS_PER_JOB[j] and not job_done[j]:
                    job_done[j]  = True
                    job_times[j] = job_sim_time[j]
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) finished'
                          f' at round {job_rounds[j]}'
                          f' — acc={acc:.2f}%'
                          f' (sim time: {job_times[j]:.1f} min)')

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(NUM_JOBS) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    os.makedirs('results', exist_ok=True)
    with open('results/bods_log.json', 'w') as f:
        json.dump(log, f)
    print('  Log saved -> results/bods_log.json')

    total_time = sum(job_times.values())

    print('\n' + '=' * 70)
    print('BODS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<26} {"Sim Time (min)":>14} {"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(NUM_JOBS):
        print(f'  {j}    {JOB_NAMES[j]:<26} '
              f'{job_times[j]:>13.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
    print('-' * 70)
    print(f'  Total simulated time: {total_time:.1f} min')

    print('\nFairness (sigma of per-device selection counts):')
    for j in range(NUM_JOBS):
        stats = agents[j].get_fairness_stats()
        print(f'  Job {j}: sigma={stats["std"]:.2f}  '
              f'min={stats["min"]}  max={stats["max"]}')
    print('=' * 70)

    print('\nPaper expected (Table 2, non-IID):')
    print('  ResNet: 58.3%  CNN-B: 83.6%  AlexNet: 99.0%')
    print('=' * 70)


if __name__ == '__main__':
    main()