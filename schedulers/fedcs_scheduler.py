"""
Multi-Job Federated Learning — FedCS Scheduler
===============================================
Reproduces the experimental setup of:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

FedCS (Nishio & Yonetani, ICC 2019):
  - Devices report capability (computation + communication)
  - Server selects devices with capability-weighted probability
  - Higher capability = higher chance of being selected
  - Occupied devices excluded each round

Setup (matching paper Table 4 / Section 5):
  - 100 devices total
  - 10 devices selected per round per job
  - Non-IID partition: 2 classes per device
  - Local training: 5 epochs
  - Batch sizes: ResNet=30, CNN-B=10, AlexNet=64
  - Learning rates: ResNet=0.1, CNN-B=0.01, AlexNet=0.01
  - FedAvg aggregation

Jobs (Group B, paper Table 4):
  Job 0 : ResNet-18  + CIFAR-10       target 54.6%
  Job 1 : CNN-B      + Fashion-MNIST  target 82.1%
  Job 2 : AlexNet    + MNIST          target 98.9%
"""
import os
import sys
import time
import random
import json
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resnet  import ResNet18
from models.cnn_b   import CNNB
from models.alexnet import AlexNet
from federated.client import FLClient
from federated.server import FLServer
from models.non_iid_partition import create_non_iid_datasets

# ── Live plot support ─────────────────────────────────────────────────────────
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
    _fig.suptitle('FedCS Scheduler — Training Progress', fontsize=13)
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
    if _fig is None:
        return
    os.makedirs('results', exist_ok=True)
    path = 'results/fedcs_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Plot saved -> {path}')


# ── FedCS Device Selector ─────────────────────────────────────────────────────
class FedCSSelector:
    """
    FedCS selects devices using capability-weighted probability sampling.
    Higher capability devices are more likely to be selected but not
    deterministically always the same ones — this matches the deadline
    mechanism of the original FedCS paper where capable devices are
    preferred but variety is maintained across rounds.
    """
    def __init__(self, num_devices, devices_per_round, device_capabilities, seed=42):
        self.num_devices       = num_devices
        self.devices_per_round = devices_per_round
        self.capability = np.array([
            device_capabilities[d]['capability']
            for d in range(num_devices)
        ])
        self.selection_counts = {j: np.zeros(num_devices) for j in range(3)}
        self.rng = np.random.default_rng(seed)

    def select(self, job_id, occupied):
        available = [d for d in range(self.num_devices) if d not in occupied]
        if len(available) == 0:
            return []

        # Capability-weighted probability — higher capability = higher chance
        caps = self.capability[available]
        probs = caps / caps.sum()

        n = min(self.devices_per_round, len(available))
        selected = self.rng.choice(
            available,
            size=n,
            replace=False,
            p=probs
        ).tolist()

        for d in selected:
            self.selection_counts[job_id][d] += 1
        return selected

    def get_stats(self):
        stats = {}
        for j in range(3):
            counts = self.selection_counts[j]
            stats[j] = {
                'std': float(np.std(counts)),
                'min': int(np.min(counts)),
                'max': int(np.max(counts)),
            }
        return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('MULTI-JOB FEDCS SCHEDULER  (paper: Zhou et al. AAAI-22)')
    print('=' * 70)

    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice : {device}')

    NUM_DEVICES       = 100
    DEVICES_PER_ROUND = 10
    LOCAL_EPOCHS      = 5
    BATCH_SIZE        = {0: 30, 1: 10, 2: 64}
    LEARNING_RATE     = {0: 0.1, 1: 0.01, 2: 0.01}
    MAX_ROUNDS        = 5000

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

    print('\nJobs:')
    for j, (m, d, _, _, t) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  target={t}%')

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
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  params={n_params:,}')

    # ── Device capabilities ───────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    device_caps = {
        d: {
            'capability':  float(rng.uniform(0.5, 2.0)),
            'fluctuation': float(rng.uniform(0.1, 1.0)),
        }
        for d in range(NUM_DEVICES)
    }

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    for j in range(3):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d],
                        BATCH_SIZE[j], LEARNING_RATE[j], device)
            for d in range(NUM_DEVICES)
        }
        print(f'  Job {j}: {NUM_DEVICES} clients created')

    # ── FedCS Selector ────────────────────────────────────────────────────────
    print('\nInitialising FedCS selector...')
    selector = FedCSSelector(NUM_DEVICES, DEVICES_PER_ROUND, device_caps, seed=SEED)

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()

    log           = {j: {'rounds': [], 'acc': [], 'loss': []} for j in range(3)}
    job_done      = {0: False, 1: False, 2: False}
    job_times     = {0: 0.0,   1: 0.0,   2: 0.0}
    job_rounds    = {0: 0,     1: 0,     2: 0}
    job_final_acc = {0: 0.0,   1: 0.0,   2: 0.0}
    round_num     = 0
    global_start  = time.time()

    with tqdm(total=MAX_ROUNDS, desc='FedCS') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1
            occupied = set()

            for j in range(3):
                if job_done[j]:
                    continue

                selected = selector.select(j, occupied)
                occupied.update(selected)

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

                update_plots(j, job_rounds[j], acc, loss)

                log[j]['rounds'].append(job_rounds[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                target = JOBS[j][4]
                if acc >= target and not job_done[j]:
                    job_done[j] = True
                    job_times[j] = time.time() - global_start
                    print(f'\n  ✓ Job {j} ({JOB_NAMES[j]}) reached {target}%'
                          f' at round {job_rounds[j]}'
                          f' ({job_times[j]/60:.1f} min)')

            pbar.update(1)
            if round_num % 20 == 0:
                status = [f'J{j}:{job_final_acc[j]:.1f}%'
                          for j in range(3) if not job_done[j]]
                if status:
                    pbar.set_postfix_str(' '.join(status))

    save_plots()

    os.makedirs('results', exist_ok=True)
    log_path = 'results/fedcs_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f)
    print(f'  Log saved -> {log_path}')

    stats      = selector.get_stats()
    total_time = sum(job_times.values())

    print('\n' + '=' * 70)
    print('FEDCS RESULTS')
    print('=' * 70)
    print(f'{"Job":<6} {"Model+Dataset":<26} {"Time (min)":>12} {"Rounds":>8} {"Acc":>8}')
    print('-' * 70)
    for j in range(3):
        print(f'  {j}    {JOB_NAMES[j]:<26} '
              f'{job_times[j]/60:>10.1f}   '
              f'{job_rounds[j]:>6}   '
              f'{job_final_acc[j]:>6.2f}%')
    print('-' * 70)
    print(f'  Total time: {total_time/60:.1f} min  ({total_time/3600:.2f} h)')

    print('\nFairness (sigma of per-device selection counts):')
    for j in range(3):
        print(f'  Job {j}: sigma={stats[j]["std"]:.2f}  '
              f'min={stats[j]["min"]}  max={stats[j]["max"]}')
    print('=' * 70)


if __name__ == '__main__':
    main()