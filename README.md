# AAMS: Adaptive Asynchronous Multi-Job Federated Learning Scheduling

Code for the paper:
**Adaptive Asynchronous Multi-Job Federated Learning Scheduling for Heterogeneous Medical Imaging Tasks**
Submitted to ARRL-Health @ IEEE BIBM 2026.

## Overview

AAMS is a device scheduling algorithm for multi-job federated learning that jointly addresses:
- C1: Synchronous aggregation bottleneck
- C2: Fixed aggregation thresholds under dynamic convergence
- C3: Wasted straggler computation across concurrent jobs

## Requirements

Install dependencies:
    pip install torch torchvision medmnist numpy tqdm scikit-learn matplotlib

## Repository Structure

    schedulers/
      aams_scheduler.py           AAMS (general benchmarks)
      aams_medical.py             AAMS (medical imaging)
      aams_medical_dropout.py     AAMS with device dropout
      rlds_scheduler.py           RLDS baseline
      rlds_medical.py             RLDS (medical)
      bods_scheduler.py           BODS baseline
      bods_medical.py             BODS (medical)
      fajs_scheduler.py           FairFedJS baseline
      fajs_medical.py             FairFedJS (medical)
      jtsra_scheduler.py          JTSRA baseline
      jtsra_medical.py            JTSRA (medical)
    models/
      resnet.py                   ResNet-18
      medical_models.py           ResNet-18 for MedMNIST
      non_iid_partition.py        Non-IID data partitioning
    federated/
      client.py
      server.py
    utils/
      time_simulator.py           Device time simulation

## Running Experiments

General benchmarks:
    python schedulers/aams_scheduler.py
    python schedulers/rlds_scheduler.py

Medical imaging (set NUM_JOBS = 2, 3, or 4 inside file):
    python schedulers/aams_medical.py
    python schedulers/rlds_medical.py
    python schedulers/fajs_medical.py
    python schedulers/jtsra_medical.py

Device dropout experiment (set DROPOUT_RATE = 0.1, 0.2, or 0.3 inside file):
    python schedulers/aams_medical_dropout.py

## Key Hyperparameters

NUM_DEVICES : 200   Total federated devices
NUM_JOBS    : 3     Concurrent FL jobs
K           : 10    Devices per job per round
BETA_MIN    : 0.5   Lower bound for adaptive K_min
BETA_MAX    : 0.7   Upper bound for adaptive K_min
W           : 10    Convergence rate window size

## Results

Logs saved to results/ as .log files.
Per-round accuracy and simulated time saved as *_log.json.
