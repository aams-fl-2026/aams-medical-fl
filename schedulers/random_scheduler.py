#!/usr/bin/env python
"""
Multi-Job Federated Learning — Random Scheduler Baseline
=========================================================
Reproduces the experimental setup of:
  "Efficient Device Scheduling with Multi-Job Federated Learning"
  Zhou et al., AAAI 2022  (arXiv:2112.05928)

Setup (matching paper Table 1 / Section 5):
  - 100 devices total
  - 10 devices selected per round per job
  - Non-IID partition: 2 classes per device (Dirichlet-like)
  - Local training: 5 epochs, batch size 64, SGD lr=0.01
  - FedAvg aggregation

Jobs (paper Section 5 experimental config):
  Job 0 : ResNet-18  + CIFAR-10  (3-ch, 32×32)  — target 80% (paper) / ~52% FL non-IID
  Job 1 : LeNet-5    + MNIST     (1-ch, 28×28)  — target 99% (paper) / ~98% FL non-IID
  Job 2 : AlexNet    + MNIST     (1-ch, 28×28)  — target 99% (paper) / ~99% FL non-IID

Target accuracies used here are the FL non-IID convergence targets that
reproduce the relative speedup results from the paper.
"""

import os
import sys
import time
import random
import json

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.resnet  import ResNet18
from models.cnn_b   import CNNB
from models.alexnet import AlexNet
from federated.client import FLClient
from federated.server import FLServer
from models.non_iid_partition import create_non_iid_datasets

# ── Live plot support (graceful fallback if display unavailable) ─────────────
try:
    import matplotlib
    matplotlib.use('Agg')          # works on headless servers / Kaggle
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ── Global plot state ────────────────────────────────────────────────────────
_fig = _axes = None
_acc_history  = {0: [], 1: [], 2: []}
_loss_history = {0: [], 1: [], 2: []}
_round_history = {0: [], 1: [], 2: []}
_JOB_COLORS = {0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green'}
_JOB_NAMES  = {0: 'Job 0', 1: 'Job 1', 2: 'Job 2'}

def setup_plots():
    global _fig, _axes
    if not PLOT_AVAILABLE:
        return
    plt.ion()
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig.suptitle('Random Scheduler — Training Progress', fontsize=13)
    for ax, title, ylabel in [
        (_axes[0], 'Test Accuracy per Round',  'Accuracy (%)'),
        (_axes[1], 'Test Loss per Round',       'Loss'),
    ]:
        ax.set_xlabel('Round'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def update_plots(job_id, round_num, acc, loss):
    """Call at the end of every round for each active job."""
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
    import os
    os.makedirs('results', exist_ok=True)
    path = 'results/random_training_curves.png'
    _fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Plot saved to {path}')


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print('=' * 70)
    print('MULTI-JOB RANDOM SCHEDULER  (paper: Zhou et al. AAAI-22)')
    print('=' * 70)

    # ── Reproducibility ──────────────────────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice : {device}')

    # ── Hyperparameters (matching paper Section 5) ───────────────────────────
    NUM_DEVICES      = 100   # total devices  (paper: 100)
    DEVICES_PER_ROUND = 10   # selected / round / job  (paper: 10)
    LOCAL_EPOCHS     = 5     # local SGD epochs  (paper: 5)
    BATCH_SIZE       = {0: 30, 1: 10, 2: 64}    # local batch size  (paper: 64)
    LEARNING_RATE    = {0: 0.1, 1: 0.01, 2: 0.01}  # SGD lr  (paper: 0.01)
    MAX_ROUNDS       = 5000   # safety cap

    # ── Job configuration (paper Section 5) ──────────────────────────────────
    # Format: model_key, dataset, in_channels, input_size, target_acc
    #
    # Target accuracies are the FL non-IID convergence thresholds that
    # replicate the paper's relative timing results.
    #   Job 0 ResNet18+CIFAR10 : paper reports ~80% standalone; FL non-IID ~52%
    #   Job 1 LeNet5+MNIST     : paper reports ~99% standalone; FL non-IID ~98%
    #   Job 2 AlexNet+MNIST    : paper reports ~99% standalone; FL non-IID ~99%
    JOBS = {
        0: ('resnet18', 'cifar10', 3, 32, 54.6),
        1: ('cnn_b',   'fashion_mnist',   1, 28, 82.1),
        2: ('alexnet',  'mnist',   1, 28, 98.9),
    }
    JOB_NAMES = {
        0: 'ResNet18 + CIFAR-10',
        1: 'CNN  + FashionMNIST',
        2: 'AlexNet  + MNIST',
    }

    print('\nJobs:')
    for j, (m, d, _, _, t) in JOBS.items():
        print(f'  Job {j}: {JOB_NAMES[j]:25s}  target={t}%')

    # ── Datasets ─────────────────────────────────────────────────────────────
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

    # ── Models & servers ─────────────────────────────────────────────────────
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

    # ── Clients ───────────────────────────────────────────────────────────────
    print(f'\nCreating {NUM_DEVICES} clients per job...')
    clients = {}
    for j in range(3):
        clients[j] = {
            d: FLClient(d, job_client_datasets[j][d], BATCH_SIZE[j], LEARNING_RATE[j], device)
            for d in range(NUM_DEVICES)
        }

    # ── Training loop ─────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    setup_plots()   # ◄── live plots initialised HERE (before loop)

    job_done   = {0: False, 1: False, 2: False}
    job_times  = {0: 0.0,   1: 0.0,   2: 0.0}
    job_rounds = {0: 0,     1: 0,     2: 0}
    job_final_acc = {0: 0.0, 1: 0.0, 2: 0.0}

    # Log for saving to disk
    log = {j: {'rounds': [], 'acc': [], 'loss': []} for j in range(3)}

    global_start = time.time()
    round_num = 0

    with tqdm(total=MAX_ROUNDS, desc='Random') as pbar:
        while not all(job_done.values()) and round_num < MAX_ROUNDS:
            round_num += 1
            occupied = set()

            for j in range(3):
                if job_done[j]:
                    continue

                # ── Device selection: uniform random (Random baseline) ────────
                available = [d for d in range(NUM_DEVICES) if d not in occupied]
                rng = random.Random(SEED + round_num * 10 + j)
                selected = rng.sample(available, min(DEVICES_PER_ROUND, len(available)))
                occupied.update(selected)

                # ── Local training + FedAvg aggregation ──────────────────────
                local_updates = [
                    clients[j][d].train(servers[j].global_model, LOCAL_EPOCHS)
                    for d in selected
                ]
                servers[j].aggregate(local_updates)
                job_rounds[j] += 1

                # ── Evaluate ──────────────────────────────────────────────────
                result = servers[j].evaluate()
                acc  = result['test_accuracy']
                loss = result['test_loss']
                job_final_acc[j] = acc

                # ── ▼▼▼ LIVE PLOT UPDATE — called every round ▼▼▼ ────────────
                update_plots(j, job_rounds[j], acc, loss)
                # ── ▲▲▲ END PLOT UPDATE ▲▲▲ ──────────────────────────────────

                log[j]['rounds'].append(job_rounds[j])
                log[j]['acc'].append(acc)
                log[j]['loss'].append(loss)

                # ── Check target ──────────────────────────────────────────────
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

    save_plots()   # ◄── save final plot

    # ── Save log ──────────────────────────────────────────────────────────────
    log_path = '/kaggle/working/random_log.json'
    try:
        with open(log_path, 'w') as f:
            json.dump(log, f)
        print(f'  Log saved → {log_path}')
    except Exception:
        pass

    # ── Results ───────────────────────────────────────────────────────────────
    total_time = sum(job_times.values())
    print('\n' + '=' * 70)
    print('RANDOM RESULTS')
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
    print('=' * 70)


if __name__ == '__main__':
    main()
