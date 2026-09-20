#!/usr/bin/env python
"""
Multi-Job Federated Learning — AAMS Scheduler
==============================================
Adaptive Asynchronous Multi-Job Scheduling (AAMS)

Novel contribution combining three mechanisms:

1. RLDS-based device selection (paper baseline)
   - Shared policy network selects fast devices per job

2. Adaptive asynchronous aggregation
   - K_min(r) adapts per job based on convergence rate
   - Jobs converging fast -> lower K_min -> faster rounds
   - Jobs struggling -> higher K_min -> more gradient updates
   - K_min(r) = ceil(K * clamp(1 - conv_rate_m, min_ratio, max_ratio))
   - When only 1 job remains: K_min = K (full synchronous, no stragglers)

3. Straggler reuse across jobs
   - Instead of dropping slow devices, assign them to
     the job with lowest recent accuracy improvement
   - Zero extra time cost since they run in parallel
   - Capped at K devices per target job per round
"""
import os
import sys
import json
import random
import math
import numpy as np
DROPOUT_RATE = 0.30  # global device dropout rate
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet            import ResNet18
from models.medical_models    import ResNet50Medical, DenseNet121Medical, VGG16Medical, DenseNet169Medical
from models.cnn_b             import CNNB
from models.alexnet           import AlexNet
from models.vgg               import VGG11
from federated.client         import FLClient
from federated.server         import FLServer
from models.non_iid_partition import create_non_iid_datasets
from utils.time_simulator     import DeviceTimeSimulator

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

NUM_JOBS = 3

_fig = _axes = None
_time_history    = {j: [] for j in range(NUM_JOBS)}
_acc_history     = {j: [] for j in range(NUM_JOBS)}
_loss_history    = {j: [] for j in range(NUM_JOBS)}
_kmin_history    = {j: [] for j in range(NUM_JOBS)}
_reward_history  = []
_REWARD_WINDOW   = 20
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
    _fig, _axes = plt.subplots(2, 2, figsize=(18, 10))
    _fig.suptitle('AAMS Scheduler — Adaptive Async Multi-Job FL', fontsize=13)
    for (r, c), title, ylabel, xlabel in [
        ((0,0), 'Test Accuracy vs Simulated Time', 'Accuracy (%)',  'Simulated Time (min)'),
        ((0,1), 'Test Loss vs Simulated Time',     'Loss',          'Simulated Time (min)'),
        ((1,0), 'Adaptive K_min per Job',          'K_min',         'Round'),
        ((1,1), 'TotalCost Reward (sliding avg)',  'Avg Reward',    'Round'),
    ]:
        _axes[r,c].set_xlabel(xlabel); _axes[r,c].set_ylabel(ylabel)
        _axes[r,c].set_title(title);   _axes[r,c].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def update_plots(job_id, sim_time, acc, loss, kmin, round_num):
    _time_history[job_id].append(sim_time)
    _acc_history[job_id].append(acc)
    _loss_history[job_id].append(loss)
    _kmin_history[job_id].append((round_num, kmin))
    if not PLOT_AVAILABLE or _axes is None:
        return
    _axes[0,0].cla(); _axes[0,1].cla(); _axes[1,0].cla()
    for j in range(NUM_JOBS):
        if _time_history[j]:
            _axes[0,0].plot(_time_history[j], _acc_history[j],
                            color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
            _axes[0,1].plot(_time_history[j], _loss_history[j],
                            color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
        if _kmin_history[j]:
            rounds, kmins = zip(*_kmin_history[j])
            _axes[1,0].plot(rounds, kmins, color=_JOB_COLORS[j],
                            label=_JOB_NAMES[j], lw=1.5)
    for ax, ylabel in [(_axes[0,0], 'Accuracy (%)'), (_axes[0,1], 'Loss')]:
        ax.set_xlabel('Simulated Time (min)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _axes[1,0].set_xlabel('Round'); _axes[1,0].set_ylabel('K_min')
    _axes[1,0].set_title('Adaptive K_min per Job over Rounds')
    _axes[1,0].grid(True, alpha=0.3); _axes[1,0].legend(fontsize=8)
    _fig.canvas.draw(); _fig.canvas.flush_events()


def update_reward_plot(reward):
    _reward_history.append(reward)
    if not PLOT_AVAILABLE or _axes is None:
        return
    if len(_reward_history) >= _REWARD_WINDOW:
        smoothed = [np.mean(_reward_history[max(0,i-_REWARD_WINDOW):i+1])
                    for i in range(len(_reward_history))]
        _axes[1,1].cla()
        _axes[1,1].plot(smoothed, color='tab:red', lw=1.5,
                        label=f'Sliding avg (w={_REWARD_WINDOW})')
        _axes[1,1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        _axes[1,1].set_xlabel('Round'); _axes[1,1].set_ylabel('Avg Reward')
        _axes[1,1].set_title(f'TotalCost Reward (w={_REWARD_WINDOW})')
        _axes[1,1].grid(True, alpha=0.3); _axes[1,1].legend(fontsize=8)
    _fig.canvas.draw(); _fig.canvas.flush_events()


def save_plots():
    if not PLOT_AVAILABLE or _fig is None:
        return
    os.makedirs('results', exist_ok=True)
    _fig.savefig('results/aams_training_curves.png', dpi=150, bbox_inches='tight')
    print('  Plot saved -> results/aams_training_curves.png')
    if _reward_history:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        smoothed = [np.mean(_reward_history[max(0,i-_REWARD_WINDOW):i+1])
                    for i in range(len(_reward_history))]
        ax2.plot(_reward_history, color='lightcoral', lw=0.8, alpha=0.5, label='Raw')
        ax2.plot(smoothed, color='tab:red', lw=2.0,
                 label=f'Sliding avg (w={_REWARD_WINDOW})')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Round'); ax2.set_ylabel('Reward')
        ax2.set_title('AAMS TotalCost Reward Curve')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig('results/aams_reward_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print('  Reward curve saved -> results/aams_reward_curve.png')


class RLDSPolicyNetwork(nn.Module):
    def __init__(self, num_devices, num_jobs=3, hidden_size=128):
        super(RLDSPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(2 * num_devices + num_jobs, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_devices)

    def forward(self, state):
        out, _ = self.lstm(state)
        return torch.softmax(self.fc(out[:, -1, :]), dim=-1).squeeze(0)


class AAMSScheduler:
    def __init__(self, num_devices, devices_per_round, num_jobs,
                 device_caps, alpha=0.7, beta=0.3,
                 min_ratio=0.5, max_ratio=0.7,
                 conv_window=10,
                 epsilon_greedy=0.3, epsilon_decay=0.999,
                 gamma=0.9, lr=1e-3, hidden_size=128,
                 torch_device='cpu'):

        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.num_jobs          = num_jobs
        self.device_caps       = device_caps
        self.alpha             = alpha
        self.beta              = beta
        self.min_ratio         = min_ratio
        self.max_ratio         = max_ratio
        self.conv_window       = conv_window
        self.epsilon_greedy    = epsilon_greedy
        self.epsilon_decay     = epsilon_decay
        self.gamma             = gamma
        self.torch_device      = torch_device

        self.acc_history     = {j: [] for j in range(num_jobs)}
        self.selection_freq  = np.zeros((num_devices, num_jobs))
        self.baselines       = np.zeros(num_jobs)
        self.round_num       = 0
        self.last_selections = {j: [] for j in range(num_jobs)}

        self.policy_net = RLDSPolicyNetwork(
            num_devices, num_jobs, hidden_size).to(torch_device)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.cap_array   = np.array([device_caps[d]['capability']
                                     for d in range(num_devices)])
        self.fluct_array = np.array([device_caps[d]['fluctuation']
                                     for d in range(num_devices)])

    def update_acc(self, job_id, acc):
        self.acc_history[job_id].append(acc)

    def get_convergence_rate(self, job_id):
        h = self.acc_history[job_id]
        if len(h) < self.conv_window:
            return 0.0
        improvement = h[-1] - h[-self.conv_window]
        return max(0.0, min(1.0, improvement / self.conv_window))

    def get_adaptive_kmin(self, job_id, num_active_jobs=2):
        """
        Adaptive K_min based on convergence rate.
        Key fix: when only 1 job remains active, use all K devices
        (full synchronous mode) so the last job converges efficiently.
        """
        # When last remaining job: use all devices, no stragglers
        if num_active_jobs <= 1:
            return self.devices_per_round

        conv_rate = self.get_convergence_rate(job_id)
        ratio = np.clip(1.0 - conv_rate * 5, self.min_ratio, self.max_ratio)
        return max(1, math.ceil(self.devices_per_round * ratio))

    def _build_state(self):
        per_device = np.empty(2 * self.num_devices)
        per_device[0::2] = self.cap_array
        per_device[1::2] = self.fluct_array
        fairness = np.array([self._fairness_cost(j) for j in range(self.num_jobs)])
        state = np.concatenate([per_device, fairness])
        return torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.torch_device)

    def _time_cost_kmin(self, selected, k_min, dropout_rate=0.0):
        times = {}
        for d in selected:
            # Device dropout: simulate device going offline mid-round
            if np.random.random() < dropout_rate:
                continue  # device dropped
            a_k  = self.device_caps[d]['capability']
            mu_k = self.device_caps[d]['fluctuation']
            times[d] = a_k + np.random.exponential(1.0 / mu_k)

        # If too few devices survived dropout, pad with slow times
        if len(times) < k_min:
            k_min = max(1, len(times))

        sorted_devices    = sorted(times.keys(), key=lambda d: times[d])
        fast_devices      = sorted_devices[:k_min]
        straggler_devices = sorted_devices[k_min:]
        round_time        = times[fast_devices[-1]] if fast_devices else 0.0

        return round_time, fast_devices, straggler_devices, times


    def _fairness_cost(self, j):
        counts = self.selection_freq[:, j]
        raw    = float(np.mean((counts - np.mean(counts)) ** 2))
        return raw / max(1, self.round_num)

    def _total_cost(self, selections, num_active=2):
        total = 0.0
        for j, selected in selections.items():
            if selected:
                k_min = self.get_adaptive_kmin(j, num_active)
                rt, _, _, _ = self._time_cost_kmin(selected, k_min, DROPOUT_RATE)
                total += self.alpha * rt + self.beta * self._fairness_cost(j)
        return total

    def select_all_jobs(self, occupied, job_done):
        state = self._build_state()
        with torch.no_grad():
            probs = self.policy_net(state).cpu().numpy()

        selections = {}
        for j in range(self.num_jobs):
            if job_done[j]:
                selections[j] = []
                self.last_selections[j] = []
                continue

            available = [d for d in range(self.num_devices) if d not in occupied]
            if not available:
                selections[j] = []
                self.last_selections[j] = []
                continue

            n = min(self.devices_per_round, len(available))
            avail_probs = np.clip(probs[available], 1e-8, None)
            avail_probs /= avail_probs.sum()

            if np.random.random() < self.epsilon_greedy:
                selected = np.random.choice(available, size=n, replace=False).tolist()
            else:
                selected = np.random.choice(available, size=n,
                                            replace=False, p=avail_probs).tolist()
            selections[j] = selected
            self.last_selections[j] = selected
            occupied.update(selected)

        return selections

    def simulate_and_reallocate(self, selections, job_done, clients, num_active):
        round_times    = {}
        fast_devs      = {}
        straggler_devs = {}
        kmin_per_job   = {}

        for j, selected in selections.items():
            if not selected or job_done[j]:
                round_times[j]    = 0.0
                fast_devs[j]      = []
                straggler_devs[j] = []
                kmin_per_job[j]   = self.devices_per_round
                continue

            k_min = self.get_adaptive_kmin(j, num_active)
            rt, fast, stragglers, _ = self._time_cost_kmin(selected, k_min, DROPOUT_RATE)

            round_times[j]    = rt
            fast_devs[j]      = fast
            straggler_devs[j] = stragglers
            kmin_per_job[j]   = k_min

        # Straggler reuse — only when multiple jobs active
        extra_devs = {j: [] for j in range(self.num_jobs)}

        active_jobs = [j for j in range(self.num_jobs)
                       if not job_done[j] and selections.get(j)]

        if len(active_jobs) >= 2:
            conv_rates   = {j: self.get_convergence_rate(j) for j in active_jobs}
            extra_counts = {j: 0 for j in active_jobs}

            for src_job in active_jobs:
                stragglers = straggler_devs.get(src_job, [])
                if not stragglers:
                    continue

                # Find most struggling job that still has room for extras
                other_jobs = [j for j in active_jobs
                              if j != src_job
                              and extra_counts[j] < self.devices_per_round]
                if not other_jobs:
                    continue

                other_jobs.sort(key=lambda j: conv_rates[j])
                target_job = other_jobs[0]

                # Assign least-used stragglers to target job
                remaining = self.devices_per_round - extra_counts[target_job]
                stragglers_sorted = sorted(
                    [d for d in stragglers if d in clients[target_job]],
                    key=lambda d: self.selection_freq[d, target_job]
                )
                assigned = stragglers_sorted[:remaining]
                for d in assigned:
                    extra_devs[target_job].append(d)
                extra_counts[target_job] += len(assigned)

        return round_times, fast_devs, extra_devs, kmin_per_job

    def update_freq(self, selections, extra_devs=None):
        for j, selected in selections.items():
            for d in selected:
                self.selection_freq[d, j] += 1
        if extra_devs:
            for j, devs in extra_devs.items():
                for d in devs:
                    self.selection_freq[d, j] += 0.5
        self.round_num += 1

    def update_policy(self, total_cost):
        reward     = -total_cost
        total_loss = None
        state = self._build_state()
        probs = self.policy_net(state)

        for j in range(self.num_jobs):
            selected = self.last_selections[j]
            if not selected:
                continue
            sel_t     = torch.LongTensor(selected).to(self.torch_device)
            log_probs = torch.log(probs[sel_t] + 1e-8)
            advantage = reward - float(self.baselines[j])
            loss_j    = -log_probs.sum() * advantage
            total_loss = loss_j if total_loss is None else total_loss + loss_j
            self.baselines[j] = ((1 - self.gamma) * self.baselines[j] +
                                  self.gamma * reward)

        if total_loss is not None:
            self.optimiser.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimiser.step()

    def decay_epsilon(self):
        self.epsilon_greedy = max(0.05, self.epsilon_greedy * self.epsilon_decay)

    def get_fairness_stats(self):
        stats = {}
        for j in range(self.num_jobs):
            c = self.selection_freq[:, j]
            stats[j] = {'std': float(np.std(c)),
                        'min': int(np.min(c)),
                        'max': int(np.max(c))}
        return stats


def pretrain(scheduler, num_rounds=50, N=5):
    print(f'  Pre-training shared policy ({num_rounds} rounds, N={N})...')
    num_active = scheduler.num_jobs
    for r in range(num_rounds):
        occupied   = set()
        selections = {}
        for j in range(scheduler.num_jobs):
            available = [d for d in range(scheduler.num_devices) if d not in occupied]
            n         = min(scheduler.devices_per_round, len(available))
            plans     = [np.random.choice(available, size=n, replace=False).tolist()
                         for _ in range(N)]
            costs     = [scheduler._total_cost({j: p}, num_active) for p in plans]
            best      = plans[np.argmin(costs)]
            selections[j]                = best
            scheduler.last_selections[j] = best
            occupied.update(best)

        scheduler.update_freq(selections)
        total_cost = scheduler._total_cost(selections, num_active)
        scheduler.update_policy(total_cost)
    print('  Pre-training complete.')


def main():
    SEED      = 42
    ALPHA     = 0.7
    BETA      = 0.3
    MIN_RATIO = 0.5
    MAX_RATIO = 0.7
    CONV_WIN  = 10

    NUM_DEVICES       = 200
    DEVICES_PER_ROUND = 10
    LOCAL_EPOCHS      = 5
    MAX_ROUNDS        = 5000

    JOBS = {
        0: ('resnet50med',    'organamnist',  1,  28, 78.0,  4),
        1: ('densenet121med', 'bloodmnist',   3,  28, 80.0,  3),
        2: ('vgg16med',       'tissuemnist',  1,  28, 50.0,  3),
###        3: ('densenet169med', 'dermamnist',   3,  28, 70.0,  7),
    }
    JOB_NAMES = {
        0: 'ResNet-50 + OrganAMNIST',
        1: 'DenseNet-121 + BloodMNIST',
        2: 'VGG-16 + TissueMNIST',
        3: 'DenseNet-169 + DermaMNIST',
    }
    NUM_JOBS = len(JOBS)

    BATCH_SIZE       = {0: 32,  1: 32,  2: 32,  3: 32}
    LEARNING_RATE_FL = {0: 0.001, 1: 0.001, 2: 0.001, 3: 0.001}

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('AAMS — Medical FL (OrganAMNIST, BloodMNIST, TissueMNIST, ISIC2019)')
    print('=' * 70)
    print(f'Device: {device}  alpha={ALPHA}  beta={BETA}')
    print(f'Adaptive K_min: ratio in [{MIN_RATIO}, {MAX_RATIO}]')
    print(f'Convergence window: {CONV_WIN} rounds')
    print(f'Straggler reuse: enabled (to most struggling job, capped per round)')
    for j, (m, d, _, _, t, _) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  target={t}%')

    print('\nCreating non-IID datasets...')
    job_client_datasets = {}
    job_test_datasets   = {}
    for j, (_, dataset, _, _, _, nc) in JOBS.items():
        client_data, test_data = create_non_iid_datasets(
            dataset, NUM_DEVICES, num_classes_per_device=nc, seed=SEED + j)
        job_client_datasets[j] = client_data
        job_test_datasets[j]   = test_data
        print(f'  Job {j} ({dataset}): {len(test_data)} test samples')

    print('\nInitialising models...')
    servers = {}
    for j, (model_key, _, ch, sz, _, _) in JOBS.items():
        dataset_name = JOBS[j][1]
        if model_key == 'resnet50med':
            model = ResNet50Medical(num_classes=11, input_channels=ch)
        elif model_key == 'densenet121med':
            model = DenseNet121Medical(num_classes=8, input_channels=ch)
        elif model_key == 'vgg16med':
            model = VGG16Medical(num_classes=8, input_channels=ch)
        elif model_key == 'densenet169med':
            model = DenseNet169Medical(num_classes=7, input_channels=ch)
        elif model_key == 'resnet18':
            num_cls = 100 if dataset_name == 'cifar100' else 10
            model = ResNet18(num_classes=num_cls, input_channels=ch)
        elif model_key == 'cnn_b':
            model = CNNB(num_classes=10)
        elif model_key == 'alexnet':
            num_cls = 100 if dataset_name == 'cifar100' else (47 if dataset_name == 'emnist_balanced' else 10)
            model = AlexNet(num_classes=num_cls, input_channels=ch, input_size=sz)
        elif model_key == 'vgg11':
            model = VGG11(num_classes=26, input_channels=ch)
        else:
            raise ValueError(f'Unknown model: {model_key}')
        servers[j] = FLServer(model.to(device), job_test_datasets[j], device=str(device))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  params={n_params:,}')

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

    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    for j in range(NUM_JOBS):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d],
                        BATCH_SIZE[j], LEARNING_RATE_FL[j], device)
            for d in range(NUM_DEVICES)
        }
        print(f'  Job {j}: {NUM_DEVICES} clients created')

    print('\nInitialising AAMS scheduler...')
    scheduler = AAMSScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=NUM_JOBS,
        device_caps=device_caps,
        alpha=ALPHA, beta=BETA,
        min_ratio=MIN_RATIO,
        max_ratio=MAX_RATIO,
        conv_window=CONV_WIN,
        epsilon_greedy=0.3,
        epsilon_decay=(0.05/0.3) ** (1.0/500),
        gamma=0.9,
        lr=1e-3,
        hidden_size=128,
        torch_device=str(device),
    )
    pretrain(scheduler, num_rounds=50, N=5)

    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()

    log                  = {j: {'sim_time': [], 'acc': [], 'loss': [], 'kmin': []} for j in range(NUM_JOBS)}
    job_done             = {j: False for j in range(NUM_JOBS)}
    job_sim_time         = {j: 0.0   for j in range(NUM_JOBS)}
    job_times            = {j: 0.0   for j in range(NUM_JOBS)}
    job_rounds           = {j: 0     for j in range(NUM_JOBS)}
    job_final_acc        = {j: 0.0   for j in range(NUM_JOBS)}
    total_extra_updates  = {j: 0     for j in range(NUM_JOBS)}
    total_stragglers     = {j: 0     for j in range(NUM_JOBS)}
    round_num            = 0

    with tqdm(total=MAX_ROUNDS, desc='AAMS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1
            occupied   = set()

            # Count active jobs this round
            num_active = sum(1 for j in range(NUM_JOBS) if not job_done[j])

            selections = scheduler.select_all_jobs(occupied, job_done)

            round_times, fast_devs, extra_devs, kmin_per_job = \
                scheduler.simulate_and_reallocate(selections, job_done, clients, num_active)

            for j in range(NUM_JOBS):
                if job_done[j]:
                    continue
                selected = selections.get(j, [])
                if not selected:
                    continue

                job_sim_time[j] += round_times[j]

                # Train fast devices
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in fast_devs.get(j, selected)
                ]

                # Train extra devices (stragglers from other jobs)
                extra = extra_devs.get(j, [])
                if extra:
                    extra_updates = [
                        clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                        for d in extra if d in clients[j]
                    ]
                    local_updates.extend(extra_updates)
                    total_extra_updates[j] += len(extra)

                # Count stragglers dropped from this job
                dropped = len(selected) - len(fast_devs.get(j, selected))
                total_stragglers[j] += dropped

                if not local_updates:
                    continue

                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                result = servers[j].evaluate()
                acc  = result['test_accuracy']
                loss = result['test_loss']
                job_final_acc[j] = acc

                scheduler.update_acc(j, acc)

                kmin = kmin_per_job.get(j, DEVICES_PER_ROUND)
                update_plots(j, job_sim_time[j], acc, loss, kmin, round_num)

                log[j]['sim_time'].append(job_sim_time[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)
                log[j]['kmin'].append(kmin)

                if acc >= JOBS[j][4] and not job_done[j]:
                    job_done[j]  = True
                    job_times[j] = job_sim_time[j]
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) reached {JOBS[j][4]}%'
                          f' at round {job_rounds[j]}'
                          f' (sim time: {job_times[j]:.1f} min)'
                          f' K_min={kmin}  extra={total_extra_updates[j]}')

            scheduler.update_freq(selections, extra_devs)
            total_cost = scheduler._total_cost(selections, num_active)
            scheduler.update_policy(total_cost)
            scheduler.decay_epsilon()
            update_reward_plot(-total_cost)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%(k={kmin_per_job.get(j,10)})'
                          for j in range(NUM_JOBS) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    os.makedirs('results', exist_ok=True)
    log['rewards'] = _reward_history
    with open('results/aams_medical_dropout_log.json', 'w') as f:
        json.dump(log, f)
    print('  Log saved -> results/aams_log.json')

    # Use actual sim time for non-converged jobs
    total_time = sum(
        job_sim_time[j] if job_times[j] == 0.0 else job_times[j]
        for j in range(NUM_JOBS)
    )
    print('\n' + '=' * 70)
    print('AAMS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<28} {"Sim Time (min)":>14} '
          f'{"Rounds":>8} {"Acc":>8} {"Extra":>8} {"Dropped":>10}')
    print('-' * 70)
    for j in range(NUM_JOBS):
        t = job_sim_time[j] if job_times[j] == 0.0 else job_times[j]
        print(f'  {j}    {JOB_NAMES[j]:<28} '
              f'{t:>13.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%'
              f'{total_extra_updates[j]:>8}'
              f'{total_stragglers[j]:>10}')
    print('-' * 70)
    print(f'  Total simulated time: {total_time:.1f} min')

    stats = scheduler.get_fairness_stats()
    print('\nFairness (sigma of per-device selection counts):')
    for j in range(NUM_JOBS):
        print(f'  Job {j}: sigma={stats[j]["std"]:.2f}  '
              f'min={stats[j]["min"]}  max={stats[j]["max"]}')
    print('=' * 70)

    if _reward_history:
        f50 = np.mean(_reward_history[:50])
        l50 = np.mean(_reward_history[-50:])
        trend = l50 - f50
        print(f'\nReward Summary:')
        print(f'  First 50 avg: {f50:.4f}')
        print(f'  Last  50 avg: {l50:.4f}')
        print(f'  Trend: {trend:+.4f} -> {"IMPROVING ✓" if trend > 0 else "NOT improving ✗"}')
    print('=' * 70)


if __name__ == '__main__':
    main()