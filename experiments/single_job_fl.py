"""
Single-Job Federated Learning

Complete FL training loop:
- Server coordinates training
- Scheduler selects devices
- Clients train locally
- Server aggregates and evaluates
"""
import sys
sys.path.append('..')

import torch
import yaml
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.non_iid_partition import create_non_iid_datasets
from models.cnn import SimpleCNN
from models.lenet import LeNet5
from federated.server import FLServer
from federated.client import FLClient
from schedulers.random_scheduler import RandomScheduler

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_model(model_name, num_classes=10):
    """Create model based on name"""
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'lenet':
        return LeNet5(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def run_federated_learning(config):
    """
    Run single-job federated learning
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dict: Training history
    """
    print("=" * 70)
    print("SINGLE-JOB FEDERATED LEARNING")
    print("=" * 70)
    
    # Extract config
    num_devices = config['system']['num_devices']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    devices_per_round = config['federated']['devices_per_round']
    
    # For this demo, use first job config
    dataset_name = config['data']['datasets']['job_0']
    model_name = config['models']['job_0']
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Model: {model_name}")
    print(f"  Devices: {num_devices}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Devices per round: {devices_per_round}")
    
    # Create datasets
    print(f"\nCreating non-IID datasets...")
    device_datasets, test_dataset = create_non_iid_datasets(
        dataset_name,
        num_devices=num_devices,
        seed=config['system']['seed']
    )
    
    # Create model
    print(f"\nCreating {model_name} model...")
    model = create_model(model_name, num_classes=10)
    
    # Create server
    print(f"\nInitializing FL Server...")
    server = FLServer(
        model,
        test_dataset,
        device=config['system']['device']
    )
    
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
    
    # Create scheduler
    print(f"\nInitializing Random Scheduler...")
    scheduler = RandomScheduler(
        num_devices=num_devices,
        devices_per_round=devices_per_round,
        seed=config['system']['seed']
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING FEDERATED TRAINING")
    print("=" * 70)
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_accuracy': [],
        'test_loss': []
    }
    
    for round_num in tqdm(range(num_rounds), desc="FL Rounds"):
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
            print(f"  Train Loss: {agg_stats['loss']:.4f}, "
                  f"Train Acc: {agg_stats['accuracy']:.2f}%")
            print(f"  Test Acc: {test_stats['test_accuracy']:.2f}%")
    
    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    final_test_acc = history['test_accuracy'][-1]
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Scheduler statistics
    print("\n" + "=" * 70)
    print("SCHEDULER STATISTICS")
    print("=" * 70)
    stats = scheduler.get_selection_stats()
    print(f"Mean selections per device: {stats['mean_selections']:.2f}")
    print(f"Fairness score (std dev): {stats['fairness_score']:.2f}")
    
    return history

def plot_results(history, save_path='results/single_job_fl.png'):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_accuracy'], label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # Load config
    config = load_config('../config.yaml')
    
    # Run FL
    history = run_federated_learning(config)
    
    # Plot results
    plot_results(history)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE!")
    print("=" * 70)