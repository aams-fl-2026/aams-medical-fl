"""
Compare Random vs FedCS Schedulers
Run both and compare results
"""
import torch
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from data.non_iid_partition import create_non_iid_datasets
from models.cnn import SimpleCNN
from federated.server import FLServer
from federated.client import FLClient
from schedulers.random_scheduler import RandomScheduler
from schedulers.fedcs_scheduler import FedCSScheduler
from utils.device_simulator import DeviceSimulator

def load_config(config_path='config.yaml'):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_fl_experiment(scheduler_name, scheduler, clients, server, num_rounds, local_epochs):
    """
    Run FL training with given scheduler
    
    Returns:
        dict: Training history
    """
    print(f"\n{'='*70}")
    print(f"RUNNING FL WITH {scheduler_name.upper()}")
    print(f"{'='*70}\n")
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'test_loss': []
    }
    
    # Reset server model (use fresh model for fair comparison)
    server.global_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    
    for round_num in tqdm(range(num_rounds), desc=f"{scheduler_name} Rounds"):
        # Select devices
        selected_device_ids = scheduler.select_devices()
        
        # Get global model
        global_model = server.get_global_model()
        
        # Each selected client trains locally
        client_updates = []
        for device_id in selected_device_ids:
            client = clients[device_id]
            update = client.train(global_model, epochs=local_epochs)
            client_updates.append(update)
        
        # Server aggregates
        agg_stats = server.aggregate(client_updates)
        
        # Evaluate on test set
        test_stats = server.evaluate()
        
        # Record history
        history['train_loss'].append(agg_stats['loss'])
        history['train_accuracy'].append(agg_stats['accuracy'])
        history['test_loss'].append(test_stats['test_loss'])
        history['test_accuracy'].append(test_stats['test_accuracy'])
        
        # Print progress every 10 rounds
        if (round_num + 1) % 10 == 0:
            print(f"\nRound {round_num + 1}/{num_rounds}:")
            print(f"  Train Loss: {agg_stats['loss']:.4f}, Train Acc: {agg_stats['accuracy']:.2f}%")
            print(f"  Test Acc: {test_stats['test_accuracy']:.2f}%")
    
    return history

def plot_comparison(random_history, fedcs_history, save_path='results/scheduler_comparison.png'):
    """Plot comparison of two schedulers"""
    
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    rounds = range(1, len(random_history['train_loss']) + 1)
    
    # Train Loss
    axes[0, 0].plot(rounds, random_history['train_loss'], label='Random', linewidth=2, marker='o', markersize=3)
    axes[0, 0].plot(rounds, fedcs_history['train_loss'], label='FedCS', linewidth=2, marker='s', markersize=3)
    axes[0, 0].set_xlabel('Round', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Train Accuracy
    axes[0, 1].plot(rounds, random_history['train_accuracy'], label='Random', linewidth=2, marker='o', markersize=3)
    axes[0, 1].plot(rounds, fedcs_history['train_accuracy'], label='FedCS', linewidth=2, marker='s', markersize=3)
    axes[0, 1].set_xlabel('Round', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0, 1].set_title('Training Accuracy', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test Loss
    axes[1, 0].plot(rounds, random_history['test_loss'], label='Random', linewidth=2, marker='o', markersize=3)
    axes[1, 0].plot(rounds, fedcs_history['test_loss'], label='FedCS', linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_xlabel('Round', fontsize=11)
    axes[1, 0].set_ylabel('Loss', fontsize=11)
    axes[1, 0].set_title('Test Loss', fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[1, 1].plot(rounds, random_history['test_accuracy'], label='Random', linewidth=2, marker='o', markersize=3)
    axes[1, 1].plot(rounds, fedcs_history['test_accuracy'], label='FedCS', linewidth=2, marker='s', markersize=3)
    axes[1, 1].set_xlabel('Round', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1, 1].set_title('Test Accuracy', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("SCHEDULER COMPARISON: RANDOM vs FEDCS")
    print("="*70)
    
    # Load config
    config = load_config('config.yaml')
    
    num_devices = config['system']['num_devices']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    devices_per_round = config['federated']['devices_per_round']
    
    print(f"\nConfiguration:")
    print(f"  Devices: {num_devices}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Devices per round: {devices_per_round}")
    
    # Create datasets
    print(f"\nCreating non-IID datasets...")
    device_datasets, test_dataset = create_non_iid_datasets(
        'cifar10',
        num_devices=num_devices,
        seed=config['system']['seed']
    )
    
    # Create device simulator
    print(f"\nInitializing device simulator...")
    simulator = DeviceSimulator(num_devices=num_devices, seed=config['system']['seed'])
    device_capabilities = simulator.get_capabilities()
    
    # Create clients
    print(f"\nCreating {num_devices} FL Clients...")
    clients = []
    for device_id in range(num_devices):
        client = FLClient(
            device_id=device_id,
            local_dataset=device_datasets[device_id],
            batch_size=config['federated']['batch_size'],
            learning_rate=config['federated']['learning_rate'],
            device=config['system']['device']
        )
        clients.append(client)
    
    # Experiment 1: Random Scheduler
    print(f"\n{'='*70}")
    print("EXPERIMENT 1: RANDOM SCHEDULER")
    print(f"{'='*70}")
    
    model_random = SimpleCNN(num_classes=10)
    server_random = FLServer(model_random, test_dataset, device=config['system']['device'])
    scheduler_random = RandomScheduler(
        num_devices=num_devices,
        devices_per_round=devices_per_round,
        seed=config['system']['seed']
    )
    
    random_history = run_fl_experiment(
        "Random",
        scheduler_random,
        clients,
        server_random,
        num_rounds,
        local_epochs
    )
    
    # Experiment 2: FedCS Scheduler
    print(f"\n{'='*70}")
    print("EXPERIMENT 2: FEDCS SCHEDULER")
    print(f"{'='*70}")
    
    model_fedcs = SimpleCNN(num_classes=10)
    server_fedcs = FLServer(model_fedcs, test_dataset, device=config['system']['device'])
    scheduler_fedcs = FedCSScheduler(
        num_devices=num_devices,
        devices_per_round=devices_per_round,
        device_capabilities=device_capabilities,
        seed=config['system']['seed']
    )
    
    fedcs_history = run_fl_experiment(
        "FedCS",
        scheduler_fedcs,
        clients,
        server_fedcs,
        num_rounds,
        local_epochs
    )
    
    # Print comparison
    print(f"\n{'='*70}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nRandom Scheduler:")
    print(f"  Final Train Accuracy: {random_history['train_accuracy'][-1]:.2f}%")
    print(f"  Final Test Accuracy: {random_history['test_accuracy'][-1]:.2f}%")
    random_stats = scheduler_random.get_selection_stats()
    print(f"  Fairness Score: {random_stats['fairness_score']:.2f}")
    print(f"  Min selections: {random_stats['min_selections']}, Max: {random_stats['max_selections']}")
    
    print(f"\nFedCS Scheduler:")
    print(f"  Final Train Accuracy: {fedcs_history['train_accuracy'][-1]:.2f}%")
    print(f"  Final Test Accuracy: {fedcs_history['test_accuracy'][-1]:.2f}%")
    fedcs_stats = scheduler_fedcs.get_selection_stats()
    print(f"  Fairness Score: {fedcs_stats['fairness_score']:.2f}")
    print(f"  Min selections: {fedcs_stats['min_selections']}, Max: {fedcs_stats['max_selections']}")
    
    fedcs_speed_stats = scheduler_fedcs.get_selected_device_speeds()
    print(f"  Devices never selected: {fedcs_speed_stats['num_not_selected_devices']}")
    
    # Plot comparison
    plot_comparison(random_history, fedcs_history)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*70}")
    print("\nKey Findings:")
    print("- FedCS selects faster devices → faster rounds (in theory)")
    print("- But FedCS has worse fairness → some devices never selected")
    print("- This impacts test accuracy in non-IID setting!")
    print(f"\nPlot saved: results/scheduler_comparison.png")

if __name__ == "__main__":
    main()