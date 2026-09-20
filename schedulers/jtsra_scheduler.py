#!/usr/bin/env python
"""
Multi-Job Federated Learning — JTSRA Scheduler (Sun et al., IEEE TWC 2025)
============================================================================
Implements the user-scheduling component from:
  H. Sun, M. Chen, Z. Yang, Y. Cang, Y. Pan, K. Huang, "Joint Task
  Scheduling and Resource Allocation for Multi-Task Federated Learning
  Over Wireless Network," IEEE Trans. Wireless Commun., vol. 25,
  pp. 9062-9077, Dec. 2025.  DOI: 10.1109/TWC.2025.3644920

The original paper jointly optimizes (a) single-slot task scheduling +
wireless resource allocation via Block Coordinate Descent (BCD) combined
with Johnson's rule, and (b) cross-slot user scheduling via a Constrained
Markov Decision Process (CMDP) solved with a Dueling Double Deep
Q-Network (D3QN) with cost shaping and prioritized experience replay.

This implementation focuses on the device-scheduling contribution that
is comparable to our simulation framework (device <-> task/job
assignment over communication rounds). The wireless physical-layer
resource-allocation sub-problem (bandwidth/power split, decided via BCD
+ Johnson's rule in the original paper) is out of scope for our
device-time simulator, which already abstracts communication+compute
time into the shift-exponential model (Formula 4) shared by all
schedulers in this codebase. We therefore implement:

  1. Single-slot task scheduling (Johnson's-rule analogue):
     Within a round, jobs are ordered by shortest-expected-processing-
     time-first (SPT), the classical two-machine flow-shop rule Johnson's
     rule reduces to for the single-resource case used here — jobs likely
     to finish fastest get first pick of available fast devices, reducing
     average round time across jobs (paper Section III).

  2. Cross-slot user scheduling via D3QN (paper Section IV):
     A Dueling Double DQN learns Q(s, a) = V(s) + [A(s,a) - mean_a A(s,a)]
     over a state capturing each job's normalized progress and each
     device's normalized speed, with actions = which device-job pairs to
     activate this round. Trained with prioritized experience replay and
     a Double-DQN target update to reduce overestimation bias, matching
     the paper's stated design (dueling head + double-Q target + PER).

  3. Constrained MDP cost-shaping (paper Section IV-B): the reward
     subtracts a penalty term scaled by how close the running energy /
     time budget is to violation, encouraging the agent to spread device
     usage instead of greedily using only the fastest devices every round
     (a softer, learned version of FAJS's explicit job-urgency rule).

This scheduler is SYNCHRONOUS like FAJS, RLDS, FedCS, BODS, Oort, and
PoC: every job still waits for all of its assigned devices to complete
before aggregating. No async K_min, no straggler reuse — both of which
remain AAMS's unique contributions.

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
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    _fig.suptitle('JTSRA Scheduler (Sun et al., IEEE TWC 2025) — Training Progress', fontsize=13)
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
    path = 'results/jtsra_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {path}')


# ── Dueling Double DQN network (paper Section IV-A) ──────────────────────────
class DuelingQNetwork(nn.Module):
    """
    Dueling architecture: Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)].
    State: per-job progress + per-job urgency + global device-speed
           histogram (coarse summary, since action space is per-job
           "how many fast/medium/slow devices to request this round").
    Action: discretized choice, per job, of how many devices from each
            speed tier {slow, medium, fast} to request this round,
            subject to a total budget of K devices/job (paper's discrete
            action space over device groups rather than raw device IDs,
            which keeps the Q-network tractable as in the paper).
    """
    def __init__(self, state_dim, num_jobs, num_tier_actions=4, hidden=128):
        super().__init__()
        self.num_jobs = num_jobs
        self.num_tier_actions = num_tier_actions   # e.g. 0..3 -> {0,3,6,10} fast devices requested
        action_dim = num_jobs * num_tier_actions

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head   = nn.Linear(hidden, action_dim)
        self.action_dim = action_dim

    def forward(self, x):
        feat = self.feature(x)
        v = self.value_head(feat)                       # (B, 1)
        a = self.adv_head(feat)                          # (B, action_dim)
        q = v + (a - a.mean(dim=1, keepdim=True))        # dueling combine
        return q   # (B, num_jobs * num_tier_actions)


# ── Prioritized Experience Replay buffer (paper Section IV-B) ───────────────
class PrioritizedReplayBuffer:
    def __init__(self, capacity=5000, alpha=0.6, eps=1e-5):
        self.capacity = capacity
        self.alpha    = alpha
        self.eps      = eps
        self.buffer   = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)

    def push(self, transition, td_error=1.0):
        self.buffer.append(transition)
        self.priorities.append((abs(td_error) + self.eps) ** self.alpha)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs / probs.sum()
        idx = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        batch = [self.buffer[i] for i in idx]
        return batch, idx

    def update_priorities(self, idx, td_errors):
        for i, e in zip(idx, td_errors):
            self.priorities[i] = (abs(e) + self.eps) ** self.alpha

    def __len__(self):
        return len(self.buffer)


# ── JTSRA Scheduler (Sun et al., IEEE TWC 2025) ──────────────────────────────
class JTSRAScheduler:
    """
    Cross-slot user scheduling via D3QN (Section IV) + single-slot task
    ordering via a Johnson's-rule analogue (Section III, "shortest
    expected processing time first" reduction for single-resource case).
    """

    DEVICE_TIERS = ['slow', 'medium', 'fast']
    TIER_REQUEST_LEVELS = [0, 3, 6, 10]   # discretized device-count actions per tier

    def __init__(self, num_devices, devices_per_round, num_jobs,
                 device_caps, job_targets, simulator=None,
                 gamma=0.95, lr=1e-3, batch_size=32,
                 target_update_every=20, energy_budget_factor=1.5,
                 torch_device='cpu', seed=42):
        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.num_jobs          = num_jobs
        self.device_caps       = device_caps
        self.job_targets       = job_targets
        self.simulator           = simulator        # DeviceTimeSimulator (Formula 4)
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_every = target_update_every
        self.energy_budget_factor = energy_budget_factor  # CMDP constraint slack
        self.torch_device       = torch_device
        self.rng                = np.random.default_rng(seed)

        # Group devices by tier for the discretized action space.
        # device_caps in this codebase only stores {'capability', 'fluctuation'}
        # (no explicit 'tier' label), so we classify by capability (a_k) using
        # the same boundaries used everywhere else to *generate* the tiers
        # (slow: a in [2,5], medium: a in [0.5,2], fast: a in [0.1,0.5]).
        self.tier_devices = {t: [] for t in self.DEVICE_TIERS}
        for d, c in device_caps.items():
            a = c['capability']
            if a >= 2.0:
                tier = 'slow'
            elif a >= 0.5:
                tier = 'medium'
            else:
                tier = 'fast'
            self.tier_devices[tier].append(d)

        # State: [progress_j, urgency_j for all jobs] + [mean speed per tier]
        self.state_dim = num_jobs * 2 + len(self.DEVICE_TIERS)
        self.num_tier_actions = len(self.TIER_REQUEST_LEVELS)

        self.q_net        = DuelingQNetwork(self.state_dim, num_jobs, self.num_tier_actions).to(torch_device)
        self.target_q_net = DuelingQNetwork(self.state_dim, num_jobs, self.num_tier_actions).to(torch_device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimiser     = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay         = PrioritizedReplayBuffer()

        self.epsilon         = 0.9
        self.epsilon_min     = 0.05
        self.epsilon_decay   = 0.998
        self.update_count    = 0

        self.progress         = np.zeros(num_jobs)
        self.selection_counts = np.zeros((num_devices, num_jobs))
        self.energy_used       = np.zeros(num_jobs)   # proxy: cumulative device-time used
        self.energy_budget      = np.full(num_jobs, 1e9)  # set after first few rounds

        self._last_state  = None
        self._last_action = None

    # ── Build state vector ───────────────────────────────────────────────────
    def _build_state(self):
        urgency = self._urgency()
        tier_speed = []
        for t in self.DEVICE_TIERS:
            ds = self.tier_devices[t]
            if not ds:
                tier_speed.append(0.0)
                continue
            speeds = [1.0 / (self.device_caps[d]['capability'] + 1.0 / self.device_caps[d]['fluctuation'])
                      for d in ds]
            tier_speed.append(float(np.mean(speeds)))
        state = list(self.progress) + list(urgency) + tier_speed
        return torch.FloatTensor(state).unsqueeze(0).to(self.torch_device)

    def _urgency(self):
        active = self.progress < 1.0
        if not active.any():
            return np.zeros(self.num_jobs)
        avg = self.progress[active].mean()
        u = np.clip(avg - self.progress, -1, 1)
        return np.where(active, u, -1.0)

    # ── Action selection: epsilon-greedy over discretized tier requests ─────
    def _select_action(self, state, job_done):
        if random.random() < self.epsilon:
            action = [random.randrange(self.num_tier_actions) for _ in range(self.num_jobs)]
        else:
            with torch.no_grad():
                q = self.q_net(state).view(self.num_jobs, self.num_tier_actions)
            action = q.argmax(dim=1).cpu().numpy().tolist()
        # Zero out requests for finished jobs
        for j in range(self.num_jobs):
            if job_done[j]:
                action[j] = 0
        return action

    # ── Convert discretized tier-request action into device IDs ─────────────
    def _action_to_devices(self, action, occupied):
        """
        Johnson's-rule analogue: order jobs by shortest expected processing
        time first (SPT) so faster-finishing jobs get first pick of the
        fastest available devices each round (paper Section III's
        single-resource reduction of the BCD+Johnson's-rule procedure).
        """
        expected_job_time = []
        for j in range(self.num_jobs):
            n_req = self.TIER_REQUEST_LEVELS[action[j]]
            if n_req == 0:
                expected_job_time.append((float('inf'), j))
                continue
            # crude expected time proxy: fewer fast-tier devices => slower
            fast_frac = len(self.tier_devices['fast']) / max(self.num_devices, 1)
            proxy_time = 1.0 / max(n_req * (0.5 + fast_frac), 1e-3)
            expected_job_time.append((proxy_time, j))
        job_order = [j for _, j in sorted(expected_job_time)]

        used = set(occupied)
        selections = {j: [] for j in range(self.num_jobs)}
        for j in job_order:
            n_req = min(self.TIER_REQUEST_LEVELS[action[j]], self.devices_per_round)
            if n_req == 0:
                continue
            # Prefer fastest available devices first (within tier priority
            # fast>medium>slow), but WITHIN each tier rank by how few times
            # this device has already been used for job j, so the same
            # handful of devices in a tier aren't selected every round
            # forever. self.tier_devices[t] is a fixed list built once at
            # init; without this count-based ordering, candidates[:n_req]
            # would always return the exact same devices every round,
            # causing severe device starvation for any job sharing a tier
            # with another active job (observed empirically as selection
            # sigma growing unboundedly with round count).
            candidates = []
            for t in ['fast', 'medium', 'slow']:
                tier_avail = [d for d in self.tier_devices[t] if d not in used]
                tier_avail.sort(key=lambda d: self.selection_counts[d, j])
                candidates.extend(tier_avail)
            chosen = candidates[:n_req]
            selections[j] = chosen
            used.update(chosen)
        return selections

    def select_devices(self, occupied, job_done):
        state = self._build_state()
        action = self._select_action(state, job_done)
        selections = self._action_to_devices(action, occupied)
        self._last_state  = state
        self._last_action = action
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

    # ── CMDP cost-shaped reward (paper Section IV-B) ─────────────────────────
    def _reward(self, selections, round_times, accuracies):
        reward = 0.0
        for j in range(self.num_jobs):
            t = round_times.get(j, 0.0)
            acc_gain = accuracies.get(j, self.progress[j] * self.job_targets[j]) \
                        - self.progress[j] * self.job_targets[j]
            base = acc_gain - 0.05 * t   # progress reward minus time penalty
            # Cost-shaping: penalize approaching the energy/time budget (CMDP constraint)
            budget_ratio = self.energy_used[j] / max(self.energy_budget[j], 1e-6)
            penalty = 0.5 * max(0.0, budget_ratio - 0.8)
            reward += base - penalty
        return reward

    # ── Update bookkeeping + store transition + train D3QN ──────────────────
    def update(self, selections, round_times, accuracies, job_done):
        for j, selected in selections.items():
            for d in selected:
                self.selection_counts[d, j] += 1
            self.energy_used[j] += round_times.get(j, 0.0) * len(selected)

        # Set adaptive energy budget after warm-up (proxy for paper's per-job
        # long-term energy constraint, Section II-C)
        for j in range(self.num_jobs):
            if self.energy_budget[j] >= 1e9 and self.energy_used[j] > 0:
                self.energy_budget[j] = self.energy_used[j] * 50 * self.energy_budget_factor

        for j, acc in accuracies.items():
            target = self.job_targets[j]
            self.progress[j] = min(acc / target, 1.0) if target > 0 else 0.0

        reward = self._reward(selections, round_times, accuracies)
        next_state = self._build_state()

        if self._last_state is not None:
            transition = (self._last_state, self._last_action, reward, next_state, all(job_done.values()))
            self.replay.push(transition, td_error=abs(reward) + 1.0)
            self._train_step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Double-DQN training step with PER ────────────────────────────────────
    def _train_step(self):
        sample = self.replay.sample(self.batch_size)
        if sample is None:
            return
        batch, idx = sample
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        rewards     = torch.FloatTensor(rewards).to(self.torch_device)
        dones       = torch.FloatTensor([float(d) for d in dones]).to(self.torch_device)

        q_values = self.q_net(states).view(-1, self.num_jobs, self.num_tier_actions)
        actions_t = torch.LongTensor(actions).to(self.torch_device)  # (B, num_jobs)
        q_sel = q_values.gather(2, actions_t.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # (B,)

        with torch.no_grad():
            # Double DQN: select action with online net, evaluate with target net
            next_q_online = self.q_net(next_states).view(-1, self.num_jobs, self.num_tier_actions)
            next_actions  = next_q_online.argmax(dim=2)   # (B, num_jobs)
            next_q_target = self.target_q_net(next_states).view(-1, self.num_jobs, self.num_tier_actions)
            next_q_sel = next_q_target.gather(2, next_actions.unsqueeze(-1)).squeeze(-1).sum(dim=1)
            target = rewards + self.gamma * next_q_sel * (1 - dones)

        td_errors = (q_sel - target).detach().cpu().numpy()
        loss = nn.functional.smooth_l1_loss(q_sel, target)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimiser.step()

        self.replay.update_priorities(idx, td_errors)

        self.update_count += 1
        if self.update_count % self.target_update_every == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

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
    MAX_ROUNDS         = 6000
    SEED               = 42

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
    print('JTSRA SCHEDULER (Sun et al., IEEE TWC 2025) — Multi-Job Federated Learning')
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

    # ── JTSRA Scheduler (D3QN) ────────────────────────────────────────────────
    print('\nInitialising JTSRA scheduler (Dueling Double DQN)...')
    scheduler = JTSRAScheduler(
        num_devices=NUM_DEVICES,
        devices_per_round=DEVICES_PER_ROUND,
        num_jobs=NUM_JOBS,
        device_caps=device_caps,
        job_targets=job_targets,
        simulator=simulator,
        gamma=0.95,
        lr=1e-3,
        batch_size=32,
        target_update_every=20,
        torch_device=str(device),
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

    with tqdm(total=MAX_ROUNDS, desc='JTSRA') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1

            # ── D3QN cross-slot scheduling + Johnson's-rule task ordering ────
            selections = scheduler.select_devices(occupied=set(), job_done=job_done)

            round_accs, round_times = {}, {}
            round_time_this_round = 0.0

            for j in range(NUM_JOBS):
                if job_done[j]:
                    round_accs[j] = JOBS[j][4]
                    continue

                selected = selections.get(j, [])
                if not selected:
                    continue

                t_round = scheduler.round_time(selected, job_id=j)
                round_times[j] = t_round
                round_time_this_round = max(round_time_this_round, t_round)

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
            scheduler.update(selections, round_times, round_accs, job_done)

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(NUM_JOBS) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status) + f' eps:{scheduler.epsilon:.2f}')

    save_plots()

    # ── Save log ──────────────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    log_path = 'results/jtsra_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f)
    print(f'  Log saved -> {log_path}')

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = sum(job_sim_time.values())
    print('\n' + '=' * 70)
    print('JTSRA RESULTS')
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