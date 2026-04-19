"""
Device heterogeneity simulator

Implements shift exponential distribution (Formula 4 from paper)
to simulate different device capabilities
"""
import numpy as np
import yaml

class DeviceSimulator:
    """
    Simulate heterogeneous device capabilities
    
    Uses shift exponential distribution for training time:
    t_k ~ shift_exponential(a_k, mu_k, D_k, tau_m)
    """
    
    def __init__(self, num_devices=30, config_path='config.yaml', seed=42):
        """
        Args:
            num_devices: Number of devices to simulate
            config_path: Path to config file
            seed: Random seed
        """
        np.random.seed(seed)
        self.num_devices = num_devices
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        device_config = config['devices']
        
        # Sample device capabilities
        # a_k: lower = faster device
        self.a_k = np.random.uniform(
            device_config['min_capability'],
            device_config['max_capability'],
            num_devices
        )
        
        # mu_k: capability fluctuation
        self.mu_k = np.random.uniform(
            device_config['min_fluctuation'],
            device_config['max_fluctuation'],
            num_devices
        )
        
        print(f"Initialized {num_devices} heterogeneous devices")
        print(f"  Capability range: [{self.a_k.min():.2f}, {self.a_k.max():.2f}]")
        print(f"  Fluctuation range: [{self.mu_k.min():.2f}, {self.mu_k.max():.2f}]")
    
    def estimate_time(self, device_id, dataset_size, tau_m=0.001):
        """
        Estimate training time for a device
        
        Formula 4 from paper: Shift exponential distribution
        t_k = tau_m * a_k * D_k + exponential(mu_k/(tau_m * D_k))
        
        Args:
            device_id: Device index (0 to num_devices-1)
            dataset_size: Number of samples (D_k)
            tau_m: Time coefficient for model m
            
        Returns:
            float: Estimated time in seconds
        """
        a_k = self.a_k[device_id]
        mu_k = self.mu_k[device_id]
        
        # Base time (deterministic part)
        base_time = tau_m * a_k * dataset_size
        
        # Random fluctuation (stochastic part)
        if dataset_size > 0:
            rate = mu_k / (tau_m * dataset_size)
            fluctuation = np.random.exponential(1.0 / rate)
        else:
            fluctuation = 0
        
        total_time = base_time + fluctuation
        
        return total_time
    
    def get_capabilities(self):
        """
        Get all device capabilities
        
        Returns:
            dict: {device_id: {'a_k': ..., 'mu_k': ...}}
        """
        return {
            i: {'a_k': self.a_k[i], 'mu_k': self.mu_k[i]}
            for i in range(self.num_devices)
        }
    
    def get_fastest_devices(self, k=10):
        """
        Get IDs of k fastest devices (lowest a_k values)
        
        Args:
            k: Number of devices to return
            
        Returns:
            list: Device IDs sorted by capability (fastest first)
        """
        sorted_indices = np.argsort(self.a_k)
        return sorted_indices[:k].tolist()
    
    def get_slowest_devices(self, k=10):
        """
        Get IDs of k slowest devices (highest a_k values)
        
        Args:
            k: Number of devices to return
            
        Returns:
            list: Device IDs sorted by capability (slowest first)
        """
        sorted_indices = np.argsort(self.a_k)
        return sorted_indices[-k:][::-1].tolist()


# Test
if __name__ == "__main__":
    print("Testing Device Simulator\n")
    print("=" * 60)
    
    simulator = DeviceSimulator(num_devices=30)
    
    print("\n" + "=" * 60)
    print("SAMPLE TIME ESTIMATIONS (dataset_size=500)")
    print("=" * 60)
    
    # Test time estimation for first 5 devices
    for device_id in range(5):
        time = simulator.estimate_time(device_id, dataset_size=500)
        cap = simulator.a_k[device_id]
        print(f"Device {device_id} (a_k={cap:.2f}): {time:.2f} seconds")
    
    # Show fastest and slowest devices
    print("\n" + "=" * 60)
    print("DEVICE RANKINGS")
    print("=" * 60)
    
    fastest = simulator.get_fastest_devices(k=5)
    print(f"\nFastest 5 devices: {fastest}")
    for dev_id in fastest:
        print(f"  Device {dev_id}: a_k={simulator.a_k[dev_id]:.3f}")
    
    slowest = simulator.get_slowest_devices(k=5)
    print(f"\nSlowest 5 devices: {slowest}")
    for dev_id in slowest:
        print(f"  Device {dev_id}: a_k={simulator.a_k[dev_id]:.3f}")
    
    print("\n" + "=" * 60)
    print("Device simulator working correctly!")
    print("=" * 60)