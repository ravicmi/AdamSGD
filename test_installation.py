"""
Test script to verify installation and setup.
Run this before executing experiments to catch any issues early.
"""

import sys
import os

def test_python_version():
    """Check Python version is 3.9 or higher."""
    print("Testing Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)")
        return False

def test_pytorch():
    """Check PyTorch installation."""
    print("Testing PyTorch...", end=" ")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        return True
    except ImportError as e:
        print(f"✗ PyTorch not found: {e}")
        return False

def test_torchvision():
    """Check torchvision installation."""
    print("Testing torchvision...", end=" ")
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
        return True
    except ImportError as e:
        print(f"✗ torchvision not found: {e}")
        return False

def test_dependencies():
    """Check all required dependencies."""
    print("Testing dependencies...", end=" ")
    required = ['numpy', 'matplotlib', 'seaborn', 'yaml', 'tqdm', 'scipy', 'pandas']
    missing = []
    
    for package in required:
        try:
            if package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print(f"✓ All dependencies installed")
        return True
    else:
        print(f"✗ Missing: {', '.join(missing)}")
        return False

def test_device():
    """Check available compute devices."""
    print("Testing compute devices...")
    try:
        import torch
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ✗ CUDA not available")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                print(f"  ✓ MPS (Apple Silicon) available")
            else:
                print(f"  ✗ MPS not available")
        
        # CPU always available
        print(f"  ✓ CPU available")
        return True
    except Exception as e:
        print(f"  ✗ Error checking devices: {e}")
        return False

def test_custom_modules():
    """Check custom modules can be imported."""
    print("Testing custom modules...", end=" ")
    try:
        from optimizers import Adam, SGDMomentum, AdaGrad, RMSProp
        from models import LogisticRegression, MLP, CNN
        from data import get_mnist_loaders, get_cifar10_loaders
        from utils import train_epoch, validate, evaluate_model
        print("✓ All custom modules importable")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Check configuration file exists and is valid."""
    print("Testing configuration...", end=" ")
    try:
        import yaml
        config_path = 'config/config.yaml'
        
        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['device', 'mode', 'adam', 'training', 'models', 'datasets']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ Missing config keys: {', '.join(missing_keys)}")
            return False
        
        print("✓ Configuration valid")
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_directories():
    """Check required directories exist."""
    print("Testing directory structure...", end=" ")
    required_dirs = [
        'config',
        'optimizers',
        'models',
        'data',
        'utils',
        'experiments',
        'results',
        'checkpoints'
    ]
    
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if not missing_dirs:
        print("✓ All directories present")
        return True
    else:
        print(f"✗ Missing directories: {', '.join(missing_dirs)}")
        return False

def test_simple_model():
    """Test creating and running a simple model."""
    print("Testing model creation and forward pass...", end=" ")
    try:
        import torch
        from models import LogisticRegression
        
        # Create model
        model = LogisticRegression(input_size=784, num_classes=10)
        
        # Create dummy input
        x = torch.randn(2, 784)
        
        # Forward pass
        output = model(x)
        
        if output.shape == (2, 10):
            print("✓ Model works correctly")
            return True
        else:
            print(f"✗ Unexpected output shape: {output.shape}")
            return False
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_optimizer():
    """Test creating and using Adam optimizer."""
    print("Testing Adam optimizer...", end=" ")
    try:
        import torch
        from models import LogisticRegression
        from optimizers import Adam
        
        # Create model
        model = LogisticRegression(input_size=784, num_classes=10)
        
        # Create optimizer
        optimizer = Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        x = torch.randn(2, 784)
        y = torch.randint(0, 10, (2,))
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("✓ Adam optimizer works correctly")
        return True
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("INSTALLATION TEST SUITE")
    print("=" * 70)
    print()
    
    tests = [
        test_python_version,
        test_pytorch,
        test_torchvision,
        test_dependencies,
        test_device,
        test_directories,
        test_config,
        test_custom_modules,
        test_simple_model,
        test_optimizer,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! You're ready to run experiments.")
        print("\nNext steps:")
        print("  1. python run_all_experiments.py")
        print("  2. Or run individual experiments from experiments/ directory")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above before running experiments.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Activate virtual environment: source adamVenv/bin/activate")
        print("  3. Check Python version: python --version (requires 3.9+)")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

