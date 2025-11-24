"""
Experiment 6.4: Advanced Experiments
Additional experiments and ablation studies from the Adam paper.
This includes experiments with different architectures and hyperparameter variations.
"""

import torch
import torch.nn as nn
import yaml
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import MLP, CNN
from data import get_mnist_loaders, get_cifar10_loaders
from optimizers import Adam, SGDMomentum, RMSProp
from utils.training import train_model
from utils.evaluation import evaluate_model
from utils.plotting import plot_training_curves, plot_comparison
import matplotlib.pyplot as plt


def get_optimizer(optimizer_name, model_parameters, config, custom_lr=None):
    """
    Get optimizer instance based on name and config.
    
    Args:
        optimizer_name (str): Name of the optimizer
        model_parameters: Model parameters to optimize
        config (dict): Configuration dictionary
        custom_lr (float, optional): Custom learning rate (for ablation studies)
    
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        lr = custom_lr if custom_lr is not None else config['adam']['lr']
        return Adam(
            model_parameters,
            lr=lr,
            betas=(config['adam']['beta1'], config['adam']['beta2']),
            eps=config['adam']['epsilon']
        )
    elif optimizer_name == 'sgd_momentum':
        lr = custom_lr if custom_lr is not None else config['sgd_momentum']['lr']
        return SGDMomentum(
            model_parameters,
            lr=lr,
            momentum=config['sgd_momentum']['momentum']
        )
    elif optimizer_name == 'rmsprop':
        lr = custom_lr if custom_lr is not None else config['rmsprop']['lr']
        return RMSProp(
            model_parameters,
            lr=lr,
            alpha=config['rmsprop']['alpha'],
            eps=config['rmsprop']['epsilon']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def run_learning_rate_ablation(config, device, train_loader, val_loader):
    """
    Run ablation study on learning rates for Adam.
    
    Args:
        config (dict): Configuration dictionary
        device: Device to train on
        train_loader: Training data loader
        val_loader: Validation data loader
    
    Returns:
        dict: Results for different learning rates
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Adam Learning Rate Sensitivity")
    print("=" * 80)
    
    # Test different learning rates
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    print(f"Testing learning rates: {learning_rates}")
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = config[config['mode']]['epochs']['advanced'] // 2  # Use fewer epochs for ablation
    
    histories = {}
    
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        
        # Create model
        model = MLP(
            input_size=config['models']['mlp']['input_size'],
            hidden_sizes=config['models']['mlp']['hidden_sizes'],
            num_classes=config['models']['mlp']['num_classes'],
            dropout=config['models']['mlp']['dropout']
        ).to(device)
        
        # Create optimizer with custom learning rate
        optimizer = get_optimizer('adam', model.parameters(), config, custom_lr=lr)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            log_interval=config['training']['log_interval']
        )
        
        histories[f'Adam (lr={lr})'] = history
    
    # Plot comparison
    save_dir = 'results/exp_6_4_advanced'
    os.makedirs(save_dir, exist_ok=True)
    
    plot_comparison(
        histories,
        metric='val_loss',
        save_path=os.path.join(save_dir, 'lr_ablation_val_loss.png'),
        config=config,
        title='Adam Learning Rate Ablation - Validation Loss',
        ylabel='Validation Loss'
    )
    
    plot_comparison(
        histories,
        metric='val_acc',
        save_path=os.path.join(save_dir, 'lr_ablation_val_acc.png'),
        config=config,
        title='Adam Learning Rate Ablation - Validation Accuracy',
        ylabel='Validation Accuracy (%)'
    )
    
    return histories


def run_beta_ablation(config, device, train_loader, val_loader):
    """
    Run ablation study on beta parameters for Adam.
    
    Args:
        config (dict): Configuration dictionary
        device: Device to train on
        train_loader: Training data loader
        val_loader: Validation data loader
    
    Returns:
        dict: Results for different beta values
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY: Adam Beta Parameter Sensitivity")
    print("=" * 80)
    
    # Test different beta combinations
    beta_configs = [
        (0.9, 0.999),   # Default (paper recommendation)
        (0.9, 0.99),    # Lower beta2
        (0.95, 0.999),  # Higher beta1
        (0.5, 0.999),   # Lower beta1
    ]
    
    criterion = nn.CrossEntropyLoss()
    num_epochs = config[config['mode']]['epochs']['advanced'] // 2
    
    histories = {}
    
    for beta1, beta2 in beta_configs:
        print(f"\nTraining with beta1={beta1}, beta2={beta2}")
        
        # Create model
        model = MLP(
            input_size=config['models']['mlp']['input_size'],
            hidden_sizes=config['models']['mlp']['hidden_sizes'],
            num_classes=config['models']['mlp']['num_classes'],
            dropout=config['models']['mlp']['dropout']
        ).to(device)
        
        # Create optimizer with custom betas
        optimizer = Adam(
            model.parameters(),
            lr=config['adam']['lr'],
            betas=(beta1, beta2),
            eps=config['adam']['epsilon']
        )
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            log_interval=config['training']['log_interval']
        )
        
        histories[f'Adam (β1={beta1}, β2={beta2})'] = history
    
    # Plot comparison
    save_dir = 'results/exp_6_4_advanced'
    
    plot_comparison(
        histories,
        metric='val_loss',
        save_path=os.path.join(save_dir, 'beta_ablation_val_loss.png'),
        config=config,
        title='Adam Beta Parameter Ablation - Validation Loss',
        ylabel='Validation Loss'
    )
    
    return histories


def run_experiment_6_4(config_path='config/config.yaml'):
    """
    Run Experiment 6.4: Advanced experiments and ablation studies.
    
    Args:
        config_path (str): Path to configuration file
    """
    print("=" * 80)
    print("EXPERIMENT 6.4: Advanced Experiments and Ablation Studies")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    torch.manual_seed(config['training']['seed'])
    
    # Setup device
    if config['device']['use_mps'] and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
    elif config['device']['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    
    # Get experiment mode
    mode = config['mode']
    print(f"Running in {mode.upper()} mode")
    
    batch_size = config[mode]['batch_size']
    num_workers = config[mode]['num_workers']
    
    # Load MNIST data for ablation studies
    print("\nLoading MNIST dataset for ablation studies...")
    train_loader, val_loader, test_loader = get_mnist_loaders(
        config, batch_size=batch_size, num_workers=num_workers
    )
    
    # Run ablation studies
    print("\n" + "=" * 80)
    print("Starting Ablation Studies")
    print("=" * 80)
    
    # 1. Learning rate ablation
    lr_histories = run_learning_rate_ablation(config, device, train_loader, val_loader)
    
    # 2. Beta parameter ablation
    beta_histories = run_beta_ablation(config, device, train_loader, val_loader)
    
    # 3. Additional experiment: Compare optimizers on deeper network
    print("\n" + "=" * 80)
    print("ADDITIONAL EXPERIMENT: Deeper Network Comparison")
    print("=" * 80)
    
    num_epochs = config[mode]['epochs']['advanced']
    criterion = nn.CrossEntropyLoss()
    
    # Create a deeper MLP
    deep_histories = {}
    optimizers_to_test = config['experiments']['exp_6_4']['optimizers']
    
    for optimizer_name in optimizers_to_test:
        print(f"\nTraining deeper MLP with {optimizer_name.upper()}")
        
        # Create deeper model (3 hidden layers instead of 2)
        model = MLP(
            input_size=784,
            hidden_sizes=[1000, 1000, 1000],  # 3 hidden layers
            num_classes=10,
            dropout=0.5
        ).to(device)
        
        optimizer = get_optimizer(optimizer_name, model.parameters(), config)
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=num_epochs,
            log_interval=config['training']['log_interval']
        )
        
        deep_histories[optimizer_name] = history
    
    # Plot deep network comparison
    save_dir = 'results/exp_6_4_advanced'
    
    plot_comparison(
        deep_histories,
        metric='train_loss',
        save_path=os.path.join(save_dir, 'deep_network_train_loss.png'),
        config=config,
        title='Deep Network (3 Hidden Layers) - Training Loss',
        ylabel='Training Loss'
    )
    
    plot_comparison(
        deep_histories,
        metric='val_acc',
        save_path=os.path.join(save_dir, 'deep_network_val_acc.png'),
        config=config,
        title='Deep Network (3 Hidden Layers) - Validation Accuracy',
        ylabel='Validation Accuracy (%)'
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 6.4 COMPLETED")
    print("=" * 80)
    print("\nGenerated ablation studies:")
    print("  1. Learning rate sensitivity analysis")
    print("  2. Beta parameter sensitivity analysis")
    print("  3. Deep network comparison")
    print(f"\nResults saved to: results/exp_6_4_advanced/")
    
    return {
        'lr_ablation': lr_histories,
        'beta_ablation': beta_histories,
        'deep_network': deep_histories
    }


if __name__ == '__main__':
    run_experiment_6_4()

