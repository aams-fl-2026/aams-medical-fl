"""
Visualize non-IID data distribution
"""
import matplotlib
matplotlib.use('Agg')  # For saving without display
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from non_iid_partition import create_non_iid_datasets
import os

def visualize_partition(num_devices=30):
    """Create visualization of data distribution"""
    
    print("Creating visualization...")
    
    # Get partitioned data
    device_datasets, _ = create_non_iid_datasets('cifar10', num_devices)
    
    # Collect class distribution for each device
    device_classes = []
    for device_id in range(num_devices):
        labels = [device_datasets[device_id][i][1] 
                  for i in range(len(device_datasets[device_id]))]
        class_dist = Counter(labels)
        device_classes.append(class_dist)
    
    # Create heatmap
    data_matrix = np.zeros((num_devices, 10))
    for device_id, class_dist in enumerate(device_classes):
        for class_id, count in class_dist.items():
            data_matrix[device_id, class_id] = count
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(data_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Number of samples')
    plt.xlabel('Class ID')
    plt.ylabel('Device ID')
    plt.title('Non-IID Data Distribution Across Devices')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results\\data_distribution.png', dpi=150)
    print("Saved visualization to results\\data_distribution.png")
    plt.close()

if __name__ == "__main__":
    visualize_partition(30)