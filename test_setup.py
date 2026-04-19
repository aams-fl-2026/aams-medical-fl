"""
Test script to verify environment setup
"""
import torch
import yaml
import sys
from pathlib import Path

def test_environment():
    print("=" * 60)
    print("TESTING ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check Python version
    print(f"\nPython version: {sys.version.split()[0]}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Test config loading
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded: {config['system']['num_devices']} devices")
    except Exception as e:
        print(f"Config error: {e}")
        return False
    
    # Test directories exist
    required_dirs = ['data', 'models', 'federated', 'schedulers', 
                     'utils', 'experiments', 'results', 'logs']
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ Directory exists: {dir_name}\\")
        else:
            print(f"✗ Missing directory: {dir_name}\\")
            all_exist = False
    
    if all_exist:
        print("\n" + "=" * 60)
        print("SETUP COMPLETE!")
        print("=" * 60)
    
    return all_exist

if __name__ == "__main__":
    test_environment()