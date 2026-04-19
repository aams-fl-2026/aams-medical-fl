"""
Multi-Job Federated Learning — RLDS Scheduler
==============================================
Implements Algorithm 2 from:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

RLDS: Reinforcement Learning-based Device Scheduling
  - LSTM policy network + fully connected layer
  - epsilon-greedy policy converter
  - Reward = -TotalCost (time cost + fairness cost)
  - Baseline EMA for variance reduction

Jobs (Group B, paper Table 4):
  Job 0 : ResNet-18  + CIFAR-10      (3-ch, 32x32, batch=30, lr=0.1)
  Job 1 : CNN-B      + Fashion-MNIST (1-ch, 28x28, batch=10, lr=0.01)
  Job 2 : AlexNet    + MNIST         (1-ch, 28x28, batch=64, lr=0.01)

Target accuracies (Table 2 non-IID convergence):
  Job 0 : 54.6%
  Job 1 : 82.1%
  Job 2 : 98.9%
"""

import os
import sys
import time
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet  import ResNet18
from models.cnn_b   import CNNB
from models.alexnet import AlexNet
from federated.client import FLClient
from federated.server import FLServer
from models.non_iid_partition import create_non_iid_datasets

# ── Plot support ─────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ── Global plot state ─────────────────────────────────────────────────────────
_fig = _axes = None
_acc_history   = {0: [], 1: [], 2: []}
_loss_history  = {0: [], 1: [], 2: []}
_round_history = {0: [], 1: [], 2: []}
_JOB_COLORS = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
_JOB_NAMES  = {0: 'ResNet18/CIFAR-10', 1: 'CNN-B/FashionMNIST', 2: 'AlexNet/MNIST'}


def setup_plots():
    global _fig, _axes
    if not PLOT_AVAILABLE:
        return
    plt.ion()
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig.suptitle('RLDS Scheduler — Training Progress', fontsize=13)
    for ax, title, ylabel in [
        (_axes[0], 'Test Accuracy per Round', 'Accuracy (%)'),
        (_axes[1], 'Test Loss per Round',      'Loss'),
    ]:
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def update_plots(job_id, round_num, acc, loss):
    _acc_history[job_id].append(acc)
    _loss_history[job_id].append(loss)
    _round_history[job_id].append(round_num)
    if not PLOT_AVAILABLE or _axes is None:
        return
    _axes[0].cla(); _axes[1].cla()
    for j in range(3):
        if _round_history[j]:
            _axes[0].plot(_round_history[j], _acc_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
            _axes[1].plot(_round_history[j], _loss_history[j],
                          color=_JOB_COLORS[j], label=_JOB_NAMES[j], lw=1.5)
    for ax, ylabel in [(_axes[0], 'Accuracy (%)'), (_axes[1], 'Loss')]:
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _axes[0].set_title('Test Accuracy per Round')
    _axes[1].set_title('Test Loss per Round')
    _fig.canvas.draw()
    _fig.canvas.flush_events()


def save_plots():
    if not PLOT_AVAILABLE or _fig is None:
        return
    plt.ioff()
    os.makedirs('results', exist_ok=True)
    path = 'results/rlds_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {path}')


# ── RLDS Policy Network (paper Section 4 — LSTM + FC) ────────────────────────
class RLDSPolicyNetwork(nn.Module):
    """
    LSTM-based policy network from paper Fig. 2.
    Input:  state vector per job = [capability, fluctuation, fairness_score]
            concatenated for all jobs -> shape (num_jobs * 3,)
    Output: probability for each device to be scheduled
    """
    def __init__(self, num_devices, num_jobs, hidden_size=64):
        super(RLDSPolicyNetwork, self).__init__()
        self.num_devices = num_devices
        self.num_jobs    = num_jobs
        input_size       = num_jobs * 3   # [cap, fluct, fairness] per job

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_devices)

    def forward(self, state):
        # state: (1, 1, input_size)
        lstm_out, _ = self.lstm(state)
        logits = self.fc(lstm_out[:, -1, :])          # (1, num_devices)
        probs  = torch.softmax(logits, dim=-1)        # (1, num_devices)
        return probs.squeeze(0)                        # (num_devices,)


# ── RLDS Scheduler (Algorithm 2) ─────────────────────────────────────────────
class RLDSScheduler:
    """
    Reinforcement Learning-based Device Scheduling (RLDS).
    Implements Algorithm 2 from Zhou et al. AAAI 2022.

    Cost model (Formula 2):
        Cost = alpha * T_r + beta * F_r
        T_r  = max execution time among selected devices (Formula 3)
        F_r  = variance of per-device selection frequency (Formula 5)
    """

    def __init__(self, num_devices, devices_per_round, num_jobs,
                 device_caps, alpha=0.5, beta=0.5,
                 epsilon=0.3, gamma=0.9, lr=1e-3,
                 torch_device='cpu'):
        self.num_devices      = num_devices
        self.devices_per_round = devices_per_round
        self.num_jobs         = num_jobs
        self.device_caps      = device_caps   # {d: {'capability': float, 'fluctuation': float}}
        self.alpha            = alpha
        self.beta             = beta
        self.epsilon          = epsilon       # epsilon-greedy exploration
        self.gamma            = gamma         # EMA decay for baseline
        self.torch_device     = torch_device

        # Selection frequency s^r_{k,m} — Formula 5
        self.selection_freq = np.zeros((num_devices, num_jobs))

        # Baseline b_m for each job (variance reduction)
        self.baselines = np.zeros(num_jobs)

        # Policy network + optimiser
        self.policy_net = RLDSPolicyNetwork(num_devices, num_jobs).to(torch_device)
        self.optimiser  = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Log probs for policy gradient update
        self.last_log_probs  = {j: None for j in range(num_jobs)}
        self.last_selections = {j: []   for j in range(num_jobs)}

    # ── State builder ────────────────────────────────────────────────────────
    def _build_state(self):
        """
        Build state vector: for each job, [mean_cap, mean_fluct, fairness].
        Returns tensor of shape (1, 1, num_jobs*3).
        """
        state = []
        for j in range(self.num_jobs):
            caps   = [self.device_caps[d]['capability']  for d in range(self.num_devices)]
            flucts = [self.device_caps[d]['fluctuation'] for d in range(self.num_devices)]
            # Fairness: std of selection frequency for job j (Formula 5)
            fairness = float(np.std(self.selection_freq[:, j]))
            state.extend([np.mean(caps), np.mean(flucts), fairness])
        t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.torch_device)
        return t

    # ── Time cost: max device execution time (Formula 3) ─────────────────────
    def _time_cost(self, selected):
        """Simulate execution time using shift-exponential model (Formula 4)."""
        times = []
        for d in selected:
            a  = self.device_caps[d]['capability']
            mu = self.device_caps[d]['fluctuation']
            # Shift-exponential: minimum time = a, then exponential tail
            shift = a
            scale = 1.0 / mu if mu > 0 else 1.0
            t = shift + np.random.exponential(scale)
            times.append(t)
        return max(times) if times else 0.0

    # ── Fairness cost: variance of selection frequency (Formula 5) ───────────
    def _fairness_cost(self, j):
        freq = self.selection_freq[:, j]
        mean = np.mean(freq)
        return float(np.mean((freq - mean) ** 2))

    # ── Total cost across all jobs (Formula 8) ────────────────────────────────
    def _total_cost(self, selections):
        total = 0.0
        for j, selected in selections.items():
            t_cost = self._time_cost(selected)
            f_cost = self._fairness_cost(j)
            total += self.alpha * t_cost + self.beta * f_cost
        return total

    # ── Device selection: epsilon-greedy policy converter ────────────────────
    def select_devices(self, occupied):
        """
        Select devices for all jobs using epsilon-greedy strategy.
        Returns dict {job_id: [device_ids]}.
        """
        state = self._build_state()
        with torch.no_grad():
            probs = self.policy_net(state)   # (num_devices,)

        selections   = {}
        log_probs    = {}
        all_occupied = set(occupied)

        for j in range(self.num_jobs):
            available = [d for d in range(self.num_devices) if d not in all_occupied]
            if len(available) == 0:
                selections[j] = []
                log_probs[j]  = None
                continue

            k = min(self.devices_per_round, len(available))

            if random.random() < self.epsilon:
                # Explore: uniform random
                selected = random.sample(available, k)
            else:
                # Exploit: sample proportional to policy probabilities
                avail_probs = probs[available].cpu().numpy()
                avail_probs = avail_probs / avail_probs.sum()   # renormalise
                selected = list(np.random.choice(
                    available, size=k, replace=False, p=avail_probs
                ))

            selections[j] = selected
            all_occupied.update(selected)

            # Compute log prob for policy gradient (Formula 12)
            sel_log_prob = torch.log(probs[selected] + 1e-8).sum()
            log_probs[j] = sel_log_prob

        self.last_log_probs  = log_probs
        self.last_selections = selections
        return selections

    # ── Update selection frequency after each round ───────────────────────────
    def update_freq(self, selections):
        for j, selected in selections.items():
            for d in selected:
                self.selection_freq[d, j] += 1

    # ── Policy gradient update (Formula 12) ──────────────────────────────────
    def update_policy(self, selections, reward_per_job):
        """
        Update policy network using REINFORCE with baseline.
        reward_per_job: dict {j: float} — reward for each job this round.
        """
        self.optimiser.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True).to(self.torch_device)

        for j in range(self.num_jobs):
            if self.last_log_probs[j] is None:
                continue
            R  = reward_per_job[j]
            bm = self.baselines[j]
            advantage = R - bm

            # Policy gradient loss: -log_prob * advantage
            loss = -self.last_log_probs[j] * advantage
            total_loss = total_loss + loss

            # Update baseline EMA
            self.baselines[j] = (1 - self.gamma) * bm + self.gamma * R

        total_loss.backward()
        self.optimiser.step()

    # ── Fairness stats ────────────────────────────────────────────────────────
    def get_fairness_stats(self):
        stats = {}
        for j in range(self.num_jobs):
            freq = self.selection_freq[:, j]
            stats[j] = {
                'std': float(np.std(freq)),
                'min': int(freq.min()),
                'max': int(freq.max()),
            }
        return stats


# ── Pre-training (Algorithm 3) ────────────────────────────────────────────────
def pretrain_policy(scheduler, num_rounds=50, N=5):
    """
    Pre-train policy network with randomly generated scheduling plans.
    Algorithm 3 from the paper.
    """
    print(f'  Pre-training policy network ({num_rounds} rounds, N={N})...')
    for r in range(num_rounds):
        occupied   = set()
        selections = {}
        for j in range(scheduler.num_jobs):
            available = [d for d in range(scheduler.num_devices) if d not in occupied]
            k = min(scheduler.devices_per_round, len(available))
            # Generate N random scheduling plans, pick best by cost
            best_sel  = None
            best_cost = float('inf')
            for _ in range(N):
                sel  = random.sample(available, k)
                cost = scheduler._time_cost(sel) + scheduler._fairness_cost(j)
                if cost < best_cost:
                    best_cost = cost
                    best_sel  = sel
            selections[j] = best_sel
            occupied.update(best_sel)

        # Compute reward and update
        total_cost = scheduler._total_cost(selections)
        reward     = -total_cost
        reward_per_job = {j: reward for j in range(scheduler.num_jobs)}

        # Build state and get log probs
        state = scheduler._build_state()
        probs = scheduler.policy_net(state)
        scheduler.last_log_probs = {}
        for j, sel in selections.items():
            if sel:
                scheduler.last_log_probs[j] = torch.log(probs[sel] + 1e-8).sum()
            else:
                scheduler.last_log_probs[j] = None

        scheduler.update_policy(selections, reward_per_job)
        scheduler.update_freq(selections)

    print('  Pre-training complete.')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Hyperparameters (matching paper Table 4 / Section 5) ─────────────────
    NUM_DEVICES      = 100
    DEVICES_PER_ROUND = 10
    LOCAL_EPOCHS     = 5
    MAX_ROUNDS       = 5000
    SEED             = 42
    ALPHA            = 0.5    # time cost weight (Formula 2)
    BETA             = 0.5    # fairness cost weight (Formula 2)

    # Per-job settings (paper Table 4)
    BATCH_SIZE    = {0: 30,   1: 10,   2: 64}
    LEARNING_RATE = {0: 0.1,  1: 0.01, 2: 0.01}

    # Jobs: (model_key, dataset, in_channels, input_size, target_acc)
    JOBS = {
        0: ('resnet18', 'cifar10',       3, 32, 54.6),
        1: ('cnn_b',    'fashion_mnist', 1, 28, 82.1),
        2: ('alexnet',  'mnist',         1, 28, 98.9),
    }

    JOB_NAMES = {
        0: 'ResNet18 + CIFAR-10',
        1: 'CNN-B  + FashionMNIST',
        2: 'AlexNet  + MNIST',
    }

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('RLDS SCHEDULER — Multi-Job Federated Learning')
    print('=' * 70)
    print(f'Device: {device}')
    print(f'Jobs:')
    for j, (m, d, _, _, t) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  target={t}%')

    # ── Datasets ──────────────────────────────────────────────────────────────
    print('\nCreating non-IID datasets (2 classes/device)...')
    job_client_datasets = {}
    job_test_datasets   = {}
    for j, (_, dataset, _, _, _) in JOBS.items():
        client_data, test_data = create_non_iid_datasets(
            dataset, NUM_DEVICES, num_classes_per_device=2, seed=SEED + j
        )
        job_client_datasets[j] = client_data
        job_test_datasets[j]   = test_data
        print(f'  Job {j} ({dataset}): {len(test_data)} test samples')

    # ── Models & servers ──────────────────────────────────────────────────────
    print('\nInitialising models...')
    servers = {}
    for j, (model_key, _, ch, sz, _) in JOBS.items():
        if model_key == 'resnet18':
            model = ResNet18(num_classes=10, input_channels=ch)
        elif model_key == 'cnn_b':
            model = CNNB(num_classes=10)
        elif model_key == 'alexnet':
            model = AlexNet(num_classes=10, input_channels=ch, input_size=sz)
        else:
            raise ValueError(f'Unknown model: {model_key}')
        servers[j] = FLServer(model.to(device), job_test_datasets[j], device=str(device))
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  params={n_params:,}')

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    rng = np.random.default_rng(SEED)
    device_caps = {
        d: {
            'capability':  float(rng.uniform(0.5, 2.0)),
            'fluctuation': float(rng.uniform(0.1, 1.0)),
        }
        for d in range(NUM_DEVICES)
    }
    clients = {}
    for j in range(3):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d],
                        BATCH_SIZE[j], LEARNING_RATE[j], device)
            for d in range(NUM_DEVICES)
        }
        for d in range(NUM_DEVICES):
            print(f'  Job {j} Client {d}: '
                  f'{len(job_client_datasets[j][d])} samples')

    # ── RLDS Scheduler ────────────────────────────────────────────────────────
    print('\nInitialising RLDS scheduler...')
    scheduler = RLDSScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=3,
        device_caps=device_caps,
        alpha=ALPHA,
        beta=BETA,
        epsilon=0.3,
        gamma=0.9,
        lr=1e-3,
        torch_device=str(device),
    )

    # Pre-train policy (Algorithm 3)
    pretrain_policy(scheduler, num_rounds=50, N=5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()

    log = {j: {'rounds': [], 'acc': [], 'loss': []} for j in range(3)}
    job_done      = {0: False, 1: False, 2: False}
    job_times     = {0: 0.0,   1: 0.0,   2: 0.0}
    job_rounds    = {0: 0,     1: 0,     2: 0}
    job_final_acc = {0: 0.0,   1: 0.0,   2: 0.0}
    round_num     = 0
    global_start  = time.time()

    with tqdm(total=MAX_ROUNDS, desc='RLDS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1

            # ── RLDS device selection ─────────────────────────────────────────
            occupied   = set()
            selections = scheduler.select_devices(occupied)

            round_accs = {}
            for j in range(3):
                if job_done[j]:
                    round_accs[j] = JOBS[j][4]
                    continue

                selected = selections.get(j, [])
                if not selected:
                    continue

                # Local training + FedAvg
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in selected
                ]
                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                # Evaluate
                result = servers[j].evaluate()
                acc    = result['test_accuracy']
                loss   = result['test_loss']
                job_final_acc[j] = acc
                round_accs[j]    = acc

                update_plots(j, job_rounds[j], acc, loss)

                log[j]['rounds'].append(job_rounds[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                # Check convergence
                target = JOBS[j][4]
                if acc >= target and not job_done[j]:
                    job_done[j]   = True
                    job_times[j]  = time.time() - global_start
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) reached {target}%'
                          f' at round {job_rounds[j]}'
                          f' ({job_times[j]/60:.1f} min)')

            # ── Update selection frequency ────────────────────────────────────
            scheduler.update_freq(selections)

            # ── Compute reward = -TotalCost ───────────────────────────────────
            total_cost = scheduler._total_cost(selections)
            reward     = -total_cost
            reward_per_job = {j: reward for j in range(3)}

            # ── Policy gradient update ────────────────────────────────────────
            scheduler.update_policy(selections, reward_per_job)

            # ── Decay epsilon (exploration -> exploitation over time) ─────────
            scheduler.epsilon = max(0.05, scheduler.epsilon * 0.9995)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(3) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    # ── Save log ──────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    log_path = 'results/rlds_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f)
    print(f'  Log saved -> {log_path}')

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = sum(job_times.values())
    print('\n' + '=' * 70)
    print('RLDS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<28} {"Time (min)":>12} '
          f'{"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(3):
        print(f'  {j}    {JOB_NAMES[j]:<28} '
              f'{job_times[j]/60:>10.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
    print('-' * 70)
    print(f'  Total time: {total_time/60:.1f} min  ({total_time/3600:.2f} h)')

    # ── Fairness stats ────────────────────────────────────────────────────────
    stats = scheduler.get_fairness_stats()
    print('\nFairness (sigma of per-device selection counts):')
    for j in range(3):
        print(f'  Job {j}: sigma={stats[j]["std"]:.2f}  '
              f'min={stats[j]["min"]}  max={stats[j]["max"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
