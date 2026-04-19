"""
Week 2 Demo for Professor Meeting
Shows all completed components
"""
import torch
print("=" * 70)
print("WEEK 2 ACCOMPLISHMENTS DEMO")
print("=" * 70)

# 1. Environment
print("\n1. ENVIRONMENT SETUP")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

# 2. Configuration
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f"\n2. CONFIGURATION")
print(f"   Devices: {config['system']['num_devices']}")
print(f"   Jobs: {config['federated']['num_jobs']}")
print(f"   Distribution: {config['data']['distribution']}")

# 3. Data
from data.non_iid_partition import create_non_iid_datasets
print(f"\n3. NON-IID DATA PARTITIONING")
print("   Creating datasets...")
device_datasets, test_dataset = create_non_iid_datasets('cifar10', num_devices=5)
print(f"   ✓ 5 devices with 2 classes each")
print(f"   ✓ Test set: {len(test_dataset)} samples")

# Show first device's class distribution
from collections import Counter
labels = [device_datasets[0][i][1] for i in range(len(device_datasets[0]))]
class_dist = Counter(labels)
print(f"   ✓ Device 0 has classes: {list(class_dist.keys())}")

# 4. Models
from models.cnn import SimpleCNN
from models.lenet import LeNet5
print(f"\n4. NEURAL NETWORK MODELS")
cnn = SimpleCNN()
lenet = LeNet5()
cnn_params = sum(p.numel() for p in cnn.parameters())
lenet_params = sum(p.numel() for p in lenet.parameters())
print(f"   ✓ SimpleCNN: {cnn_params:,} parameters")
print(f"   ✓ LeNet-5: {lenet_params:,} parameters")

# 5. Device Simulator
from utils.device_simulator import DeviceSimulator
print(f"\n5. DEVICE HETEROGENEITY SIMULATOR")
simulator = DeviceSimulator(num_devices=5)
fastest = simulator.get_fastest_devices(k=3)
slowest = simulator.get_slowest_devices(k=3)
print(f"   ✓ Fastest 3 devices: {fastest}")
print(f"   ✓ Slowest 3 devices: {slowest}")
print(f"   ✓ Fastest device capability: {simulator.a_k[fastest[0]]:.3f}")
print(f"   ✓ Slowest device capability: {simulator.a_k[slowest[0]]:.3f}")

# 6. FL Client
from federated.client import FLClient
print(f"\n6. FEDERATED LEARNING CLIENT")
client = FLClient(device_id=0, local_dataset=device_datasets[0], batch_size=32)
print(f"   ✓ Client created with {len(device_datasets[0])} samples")
print("\n   Testing training (1 epoch, ~15 seconds)...")
result = client.train(cnn, epochs=1)
print(f"   ✓ Training complete!")
print(f"   ✓ Loss: {result['loss']:.4f}")
print(f"   ✓ Accuracy: {result['accuracy']:.2f}%")
print(f"   Note: Accuracy is ~50% because device only knows 2/10 classes")

print("\n" + "=" * 70)
print("ALL COMPONENTS WORKING!")
print("=" * 70)
print("\nNext Week: FL Server, FedAvg aggregation, Random scheduler")
print("=" * 70)