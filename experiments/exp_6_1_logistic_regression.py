"""
Experiment 6.1: Logistic Regression on MNIST
Compare Adam, SGD, SGD+Momentum, AdaGrad, and RMSProp on logistic regression.
"""

import torch
import torch.nn as nn
import yaml
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import LogisticRegression
from data import get_mnist_loaders
from optimizers import Adam, SGDMomentum, AdaGrad, RMSProp
from utils.training import train_model
from utils.evaluation import evaluate_model
from utils.plotting import plot_training_curves, plot_multiple_comparisons


def get_optimizer(optimizer_name, model_parameters, config):
    """
    Get optimizer instance based on name and config.
    
    Args:
        optimizer_name (str): Name of the optimizer
        model_parameters: Model parameters to optimize
        config (dict): Configuration dictionary
    
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return Adam(
            model_parameters,
            lr=config['adam']['lr'],
            betas=(config['adam']['beta1'], config['adam']['beta2']),
            eps=config['adam']['epsilon']
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model_parameters,
            lr=config['sgd']['lr']
        )
    elif optimizer_name == 'sgd_momentum':
        return SGDMomentum(
            model_parameters,
            lr=config['sgd_momentum']['lr'],
            momentum=config['sgd_momentum']['momentum']
        )
    elif optimizer_name == 'adagrad':
        return AdaGrad(
            model_parameters,
            lr=config['adagrad']['lr'],
            eps=config['adagrad']['epsilon']
        )
    elif optimizer_name == 'rmsprop':
        return RMSProp(
            model_parameters,
            lr=config['rmsprop']['lr'],
            alpha=config['rmsprop']['alpha'],
            eps=config['rmsprop']['epsilon']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def run_experiment_6_1(config_path='config/config.yaml'):
    """
    Run Experiment 6.1: Logistic Regression on MNIST.
    
    Args:
        config_path (str): Path to configuration file
    """
    print("=" * 80)
    print("EXPERIMENT 6.1: Logistic Regression on MNIST")
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
    
    # Get number of epochs
    num_epochs = config[mode]['epochs']['logistic_regression']
    batch_size = config[mode]['batch_size']
    num_workers = config[mode]['num_workers']
    
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(
        config, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Get list of optimizers to test
    optimizers_to_test = config['experiments']['exp_6_1']['optimizers']
    print(f"\nOptimizers to test: {optimizers_to_test}")
    
    # Store results
    all_histories = {}
    test_results = {}
    
    # Train with each optimizer
    for optimizer_name in optimizers_to_test:
        print("\n" + "=" * 80)
        print(f"Training with {optimizer_name.upper()}")
        print("=" * 80)
        
        # Create model
        model = LogisticRegression(
            input_size=config['models']['logistic_regression']['input_size'],
            num_classes=config['models']['logistic_regression']['num_classes']
        ).to(device)
        
        # Create optimizer
        optimizer = get_optimizer(optimizer_name, model.parameters(), config)
        
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
        
        # Evaluate on test set
        print(f"\nEvaluating {optimizer_name} on test set...")
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        test_results[optimizer_name] = test_metrics
        
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Store history
        all_histories[optimizer_name] = history
        
        # Plot individual training curves
        save_dir = 'results/exp_6_1_logistic_regression'
        os.makedirs(save_dir, exist_ok=True)
        
        plot_training_curves(
            history,
            optimizer_name,
            save_path=os.path.join(save_dir, f'{optimizer_name}_curves.png'),
            config=config
        )
    
    # Plot comparisons
    print("\n" + "=" * 80)
    print("Generating comparison plots...")
    print("=" * 80)
    
    plot_multiple_comparisons(
        all_histories,
        save_dir='results/exp_6_1_logistic_regression',
        config=config,
        experiment_name='Exp 6.1: Logistic Regression on MNIST'
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPERIMENT 6.1 SUMMARY")
    print("=" * 80)
    print(f"{'Optimizer':<20} {'Test Loss':<12} {'Test Accuracy':<15}")
    print("-" * 80)
    for opt_name, metrics in test_results.items():
        print(f"{opt_name:<20} {metrics['loss']:<12.4f} {metrics['accuracy']:<15.2f}%")
    
    print("\nExperiment 6.1 completed!")
    print(f"Results saved to: results/exp_6_1_logistic_regression/")
    
    return all_histories, test_results


if __name__ == '__main__':
    run_experiment_6_1()

