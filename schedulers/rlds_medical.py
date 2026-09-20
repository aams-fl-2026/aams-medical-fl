#!/usr/bin/env python
"""
Multi-Job Federated Learning — RLDS Scheduler
==============================================
Implements Algorithm 2 from:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

Paper-exact:
  - ONE shared policy network (paper Figure 2)
  - State: per-device [a_k, mu_k] + fairness per job (paper Section 4)
  - Reward: R_m = -TotalCost = -sum_m[alpha*T_m + beta*F_m] (Formula 8)
  - Formula 3+4: stochastic T_m = max device time
  - Formula 5: F_m = (1/|K|)*sum_k(s_k,m - mean)^2
    normalized by round number to keep reward signal stable
  - Formula 12: policy gradient, shared network
  - Baseline EMA per job (Algorithm 2 Line 7)
  - Algorithm 3: pre-training
  - epsilon-greedy with decay

alpha > beta: paper says increase alpha for fast convergence
which gives RLDS advantage in simulated time over Random/FedCS
"""
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.medical_models    import ResNet50Medical, DenseNet121Medical, VGG16Medical, DenseNet169Medical
from models.resnet            import ResNet18
from models.cnn_b             import CNNB
from models.alexnet           import AlexNet
from models.vgg               import VGG11
from federated.client         import FLClient
from federated.server         import FLServer
from models.non_iid_partition import create_non_iid_datasets
from utils.time_simulator     import DeviceTimeSimulator


NUM_JOBS = 3

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

_fig = _axes = None
_time_history   = {j: [] for j in range(4)}
_acc_history    = {j: [] for j in range(4)}
_loss_history   = {j: [] for j in range(4)}
_reward_history = []
_REWARD_WINDOW  = 20
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
    _fig, _axes = plt.subplots(1, 3, figsize=(21, 5))
    _fig.suptitle('RLDS Scheduler — Training Progress', fontsize=13)
    for ax, title, ylabel, xlabel in [
        (_axes[0], 'Test Accuracy vs Simulated Time', 'Accuracy (%)',  'Simulated Time (min)'),
        (_axes[1], 'Test Loss vs Simulated Time',     'Loss',          'Simulated Time (min)'),
        (_axes[2], 'TotalCost Reward (sliding avg)',  'Avg Reward',    'Round'),
    ]:
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.grid(True, alpha=0.3)
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
        ax.set_xlabel('Simulated Time (min)'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _axes[0].set_title('Test Accuracy vs Simulated Time')
    _axes[1].set_title('Test Loss vs Simulated Time')
    _fig.canvas.draw(); _fig.canvas.flush_events()


def update_reward_plot(reward):
    _reward_history.append(reward)
    if not PLOT_AVAILABLE or _axes is None:
        return
    if len(_reward_history) >= _REWARD_WINDOW:
        smoothed = [np.mean(_reward_history[max(0,i-_REWARD_WINDOW):i+1])
                    for i in range(len(_reward_history))]
        _axes[2].cla()
        _axes[2].plot(smoothed, color='tab:red', lw=1.5,
                      label=f'Sliding avg (w={_REWARD_WINDOW})')
        _axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        _axes[2].set_xlabel('Round'); _axes[2].set_ylabel('Avg Reward')
        _axes[2].set_title(f'TotalCost Reward (w={_REWARD_WINDOW})')
        _axes[2].grid(True, alpha=0.3); _axes[2].legend(fontsize=8)
    _fig.canvas.draw(); _fig.canvas.flush_events()


def save_plots():
    if not PLOT_AVAILABLE or _fig is None:
        return
    os.makedirs('results', exist_ok=True)
    _fig.savefig('results/rlds_training_curves.png', dpi=150, bbox_inches='tight')
    print('  Plot saved -> results/rlds_training_curves.png')
    if _reward_history:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        smoothed = [np.mean(_reward_history[max(0,i-_REWARD_WINDOW):i+1])
                    for i in range(len(_reward_history))]
        ax2.plot(_reward_history, color='lightcoral', lw=0.8, alpha=0.5, label='Raw')
        ax2.plot(smoothed, color='tab:red', lw=2.0,
                 label=f'Sliding avg (w={_REWARD_WINDOW})')
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Round'); ax2.set_ylabel('Reward')
        ax2.set_title('RLDS TotalCost Reward Curve')
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig('results/rlds_reward_curve.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print('  Reward curve saved -> results/rlds_reward_curve.png')


# ── Shared Policy Network (paper Figure 2) ───────────────────────────────────
class RLDSPolicyNetwork(nn.Module):
    def __init__(self, num_devices, num_jobs=3, hidden_size=128):
        super(RLDSPolicyNetwork, self).__init__()
        # per-device [a_k, mu_k] + fairness per job
        self.lstm = nn.LSTM(2 * num_devices + num_jobs, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_devices)

    def forward(self, state):
        out, _ = self.lstm(state)
        return torch.softmax(self.fc(out[:, -1, :]), dim=-1).squeeze(0)


# ── RLDS Scheduler ────────────────────────────────────────────────────────────
class RLDSScheduler:
    def __init__(self, num_devices, devices_per_round, num_jobs,
                 device_caps, alpha=0.7, beta=0.3,
                 epsilon=0.3, epsilon_decay=0.9990,
                 gamma=0.9, lr=1e-3, hidden_size=128,
                 torch_device='cpu'):
        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.num_jobs          = num_jobs
        self.device_caps       = device_caps
        self.alpha             = alpha
        self.beta              = beta
        self.epsilon           = epsilon
        self.epsilon_decay     = epsilon_decay
        self.gamma             = gamma
        self.torch_device      = torch_device
        self.round_num         = 0

        # s_k,m: raw selection counts per device per job
        self.selection_freq = np.zeros((num_devices, num_jobs))
        # b_m: per-job baseline
        self.baselines = np.zeros(num_jobs)

        self.policy_net = RLDSPolicyNetwork(
            num_devices, num_jobs, hidden_size).to(torch_device)
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.cap_array   = np.array([device_caps[d]['capability']
                                     for d in range(num_devices)])
        self.fluct_array = np.array([device_caps[d]['fluctuation']
                                     for d in range(num_devices)])
        self.last_log_probs = {j: None for j in range(num_jobs)}

    def _build_state(self):
        per_device = np.empty(2 * self.num_devices)
        per_device[0::2] = self.cap_array
        per_device[1::2] = self.fluct_array
        fairness = np.array([self._fairness_cost(j) for j in range(self.num_jobs)])
        state = np.concatenate([per_device, fairness])
        return torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.torch_device)

    def _time_cost(self, selected):
        """Formula 3+4: stochastic max device time."""
        times = [self.device_caps[d]['capability'] +
                 np.random.exponential(1.0 / self.device_caps[d]['fluctuation'])
                 for d in selected]
        return max(times) if times else 0.0

    def _fairness_cost(self, j):
        """
        Formula 5: (1/|K|)*sum_k(s_k,m - mean)^2
        Normalized by round_num to keep signal stable as counts grow.
        This prevents reward from exploding while preserving the
        fairness signal the paper intends.
        """
        counts = self.selection_freq[:, j]
        raw    = float(np.mean((counts - np.mean(counts)) ** 2))
        # Normalize by round number so fairness stays bounded
        round_num = max(1, self.round_num)
        return raw / round_num

    def _total_cost(self, selections):
        """Formula 8: TotalCost = sum_m Cost_m (Formula 2)."""
        total = 0.0
        for j, selected in selections.items():
            if selected:
                total += (self.alpha * self._time_cost(selected) +
                          self.beta  * self._fairness_cost(j))
        return total

    def select_all_jobs(self, occupied):
        state = self._build_state()
        with torch.no_grad():
            probs = self.policy_net(state).cpu().numpy()

        selections = {}
        for j in range(self.num_jobs):
            available = [d for d in range(self.num_devices) if d not in occupied]
            if not available:
                selections[j] = []
                continue

            avail_probs = np.clip(probs[available], 1e-8, None)
            avail_probs /= avail_probs.sum()
            n = min(self.devices_per_round, len(available))

            if np.random.random() < self.epsilon:
                selected = np.random.choice(available, size=n, replace=False).tolist()
            else:
                selected = np.random.choice(available, size=n,
                                            replace=False, p=avail_probs).tolist()
            selections[j] = selected
            occupied.update(selected)

            sel_t      = torch.LongTensor(selected).to(self.torch_device)
            probs_grad = self.policy_net(self._build_state())
            self.last_log_probs[j] = torch.log(probs_grad[sel_t] + 1e-8)

        return selections

    def update_freq(self, selections):
        for j, selected in selections.items():
            for d in selected:
                self.selection_freq[d, j] += 1
        self.round_num += 1

    def update_policy(self, total_cost):
        """Formula 12: update shared network with TotalCost reward."""
        reward     = -total_cost
        total_loss = torch.tensor(0.0, requires_grad=True).to(self.torch_device)

        for j in range(self.num_jobs):
            if self.last_log_probs[j] is None:
                continue
            advantage  = reward - float(self.baselines[j])
            loss_j     = -self.last_log_probs[j].sum() * advantage
            total_loss = total_loss + loss_j
            self.baselines[j] = ((1 - self.gamma) * self.baselines[j] +
                                  self.gamma * reward)

        self.optimiser.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimiser.step()

    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

    def get_fairness_stats(self):
        stats = {}
        for j in range(self.num_jobs):
            c = self.selection_freq[:, j]
            stats[j] = {'std': float(np.std(c)),
                        'min': int(np.min(c)),
                        'max': int(np.max(c))}
        return stats


# ── Pre-training (Algorithm 3) ────────────────────────────────────────────────
def pretrain(scheduler, num_rounds=50, N=5):
    """Algorithm 3: pre-train with N plans per round, pick best."""
    print(f'  Pre-training shared policy ({num_rounds} rounds, N={N})...')
    for r in range(num_rounds):
        occupied   = set()
        selections = {}
        for j in range(scheduler.num_jobs):
            available = [d for d in range(scheduler.num_devices) if d not in occupied]
            n         = min(scheduler.devices_per_round, len(available))
            plans     = [np.random.choice(available, size=n, replace=False).tolist()
                         for _ in range(N)]
            costs     = [scheduler._total_cost({j: p}) for p in plans]
            best      = plans[np.argmin(costs)]
            selections[j] = best
            occupied.update(best)

        scheduler.update_freq(selections)
        total_cost = scheduler._total_cost(selections)

        state = scheduler._build_state()
        probs = scheduler.policy_net(state)
        for j, selected in selections.items():
            sel_t = torch.LongTensor(selected).to(scheduler.torch_device)
            scheduler.last_log_probs[j] = torch.log(probs[sel_t] + 1e-8)

        scheduler.update_policy(total_cost)
    print('  Pre-training complete.')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    SEED  = 42
    # alpha > beta: paper says increase alpha for fast convergence
    # this gives RLDS time advantage over Random/FedCS
    ALPHA = 0.7
    BETA  = 0.3

    NUM_DEVICES       = 100
    DEVICES_PER_ROUND = 10
    LOCAL_EPOCHS      = 5
    BATCH_SIZE       = {0: 32,  1: 32,  2: 32,  3: 32}
    LEARNING_RATE_FL = {0: 0.001, 1: 0.001, 2: 0.001, 3: 0.001}
    MAX_ROUNDS        = 5000
    N_PRETRAIN        = 5
    PRETRAIN_ROUNDS   = 50
    # Faster epsilon decay so agent exploits learned policy sooner
    EPSILON_DECAY     = (0.05/0.3) ** (1.0/500)

    JOBS = {
        0: ('resnet50med',    'organamnist',  1,  28, 78.0,  4),
        1: ('densenet121med', 'bloodmnist',   3,  28, 80.0,  3),
        2: ('vgg16med',       'tissuemnist',  1,  28, 50.0,  3),
        3: ('densenet169med', 'dermamnist',   3,  28, 70.0,  7),
    }
    JOB_NAMES = {
        0: 'ResNet-50 + OrganAMNIST',
        1: 'DenseNet-121 + BloodMNIST',
        2: 'VGG-16 + TissueMNIST',
        3: 'DenseNet-169 + DermaMNIST',
    }

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('RLDS SCHEDULER — Multi-Job Federated Learning')
    print('=' * 70)
    print(f'Device: {device}  alpha={ALPHA}  beta={BETA}')
    print(f'Shared policy  pretrain={PRETRAIN_ROUNDS}r  N={N_PRETRAIN}')
    print(f'Epsilon decay: {EPSILON_DECAY:.4f} (reaches 0.05 by ~500 rounds)')
    print('Jobs:')
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
        if model_key == 'resnet50med':
            model = ResNet50Medical(num_classes=11, input_channels=ch)
        elif model_key == 'densenet121med':
            model = DenseNet121Medical(num_classes=8, input_channels=ch)
        elif model_key == 'vgg16med':
            model = VGG16Medical(num_classes=8, input_channels=ch)
        elif model_key == 'densenet169med':
            model = DenseNet169Medical(num_classes=7, input_channels=ch)
        elif model_key == 'resnet18':
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
        print(f'  Job {j}: {JOB_NAMES[j]:30s}  params={n_params:,}')

    rng = np.random.default_rng(SEED)
    device_caps = {
        d: {'capability':  float(rng.uniform(0.5, 2.0)),
            'fluctuation': float(rng.uniform(0.1, 1.0))}
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

    print('\nInitialising RLDS scheduler (shared policy)...')
    scheduler = RLDSScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=NUM_JOBS,
        device_caps=device_caps,
        alpha=ALPHA, beta=BETA,
        epsilon=0.3,
        epsilon_decay=EPSILON_DECAY,
        gamma=0.9,
        lr=1e-3,
        hidden_size=128,
        torch_device=str(device),
    )
    pretrain(scheduler, num_rounds=PRETRAIN_ROUNDS, N=N_PRETRAIN)

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

    with tqdm(total=MAX_ROUNDS, desc='RLDS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1
            occupied   = set()
            selections = scheduler.select_all_jobs(occupied)

            for j in range(NUM_JOBS):
                if job_done[j]:
                    continue
                selected = selections.get(j, [])
                if not selected:
                    continue

                round_time = simulator.simulate_round(selected, j, LOCAL_EPOCHS)
                job_sim_time[j] += round_time

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

                log[j]['sim_time'].append(job_sim_time[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                if acc >= JOBS[j][4] and not job_done[j]:
                    job_done[j]  = True
                    job_times[j] = job_sim_time[j]
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) reached {JOBS[j][4]}%'
                          f' at round {job_rounds[j]}'
                          f' (sim time: {job_times[j]:.1f} min)')

            # Update shared network with TotalCost reward
            scheduler.update_freq(selections)
            total_cost = scheduler._total_cost(selections)
            scheduler.update_policy(total_cost)
            scheduler.decay_epsilon()
            update_reward_plot(-total_cost)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(NUM_JOBS) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    os.makedirs('results', exist_ok=True)
    log['rewards'] = _reward_history
    with open('results/rlds_medical_log.json', 'w') as f:
        json.dump(log, f)
    print('  Log saved -> results/rlds_log.json')

    total_time = sum(job_sim_time[j] if job_times[j] == 0.0 else job_times[j] for j in range(NUM_JOBS))
    print('\n' + '=' * 70)
    print('RLDS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<28} {"Sim Time (min)":>14} '
          f'{"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(NUM_JOBS):
        print(f'  {j}    {JOB_NAMES[j]:<28} '
              f'{(job_sim_time[j] if job_times[j]==0.0 else job_times[j]):>13.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
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
    print('\nPaper expected (Table 2, non-IID):')
    print('  ResNet: 53.7%  CNN-B: 82.3%  AlexNet: 98.9%')
    print('  RLDS should be fastest in simulated time')
    print('=' * 70)


if __name__ == '__main__':
    main()