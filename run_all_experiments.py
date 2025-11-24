"""
Master script to run all experiments from Section 6 of the Adam paper.
Experiments:
- 6.1: Logistic Regression on MNIST
- 6.2: Multi-Layer Perceptron on MNIST
- 6.3: Convolutional Neural Network on CIFAR-10
- 6.4: Advanced Experiments and Ablation Studies
"""

import os
import sys
import yaml
import time
from datetime import datetime
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments import (
    run_experiment_6_1,
    run_experiment_6_2,
    run_experiment_6_3,
    run_experiment_6_4
)


def create_results_directories():
    """
    Create directory structure for storing results.
    """
    directories = [
        'results',
        'results/exp_6_1_logistic_regression',
        'results/exp_6_2_mlp',
        'results/exp_6_3_cnn',
        'results/exp_6_4_advanced',
        'checkpoints'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def print_header(title):
    """
    Print formatted header.
    """
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")


def main():
    """
    Main function to run all experiments.
    """
    parser = argparse.ArgumentParser(
        description='Run all experiments from Adam paper Section 6'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=['6.1', '6.2', '6.3', '6.4'],
        help='Which experiments to run (e.g., --experiments 6.1 6.2)'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("ADAM PAPER EXPERIMENTS - SECTION 6 REPLICATION")
    print(f"Configuration file: {args.config}")
    print(f"Experiments to run: {', '.join(args.experiments)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load config to display settings
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nMode: {config['mode'].upper()}")
    print(f"Device: MPS={config['device']['use_mps']}, CUDA={config['device']['use_cuda']}")
    
    # Create results directories
    print_header("CREATING RESULTS DIRECTORIES")
    create_results_directories()
    
    # Track experiment times
    experiment_times = {}
    
    # Run experiments
    if '6.1' in args.experiments:
        print_header("EXPERIMENT 6.1: Logistic Regression on MNIST")
        start_time = time.time()
        try:
            run_experiment_6_1(args.config)
            experiment_times['6.1'] = time.time() - start_time
            print(f"\n✓ Experiment 6.1 completed in {experiment_times['6.1']:.2f} seconds")
        except Exception as e:
            print(f"\n✗ Experiment 6.1 failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    if '6.2' in args.experiments:
        print_header("EXPERIMENT 6.2: Multi-Layer Perceptron on MNIST")
        start_time = time.time()
        try:
            run_experiment_6_2(args.config)
            experiment_times['6.2'] = time.time() - start_time
            print(f"\n✓ Experiment 6.2 completed in {experiment_times['6.2']:.2f} seconds")
        except Exception as e:
            print(f"\n✗ Experiment 6.2 failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    if '6.3' in args.experiments:
        print_header("EXPERIMENT 6.3: CNN on CIFAR-10")
        start_time = time.time()
        try:
            run_experiment_6_3(args.config)
            experiment_times['6.3'] = time.time() - start_time
            print(f"\n✓ Experiment 6.3 completed in {experiment_times['6.3']:.2f} seconds")
        except Exception as e:
            print(f"\n✗ Experiment 6.3 failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    if '6.4' in args.experiments:
        print_header("EXPERIMENT 6.4: Advanced Experiments")
        start_time = time.time()
        try:
            run_experiment_6_4(args.config)
            experiment_times['6.4'] = time.time() - start_time
            print(f"\n✓ Experiment 6.4 completed in {experiment_times['6.4']:.2f} seconds")
        except Exception as e:
            print(f"\n✗ Experiment 6.4 failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print_header("ALL EXPERIMENTS COMPLETED")
    
    print("Summary:")
    print("-" * 80)
    
    total_time = sum(experiment_times.values())
    
    for exp_num, duration in experiment_times.items():
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{hours}h "
        if minutes > 0:
            time_str += f"{minutes}m "
        time_str += f"{seconds}s"
        
        print(f"Experiment {exp_num}: {time_str}")
    
    # Format total time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    total_time_str = ""
    if hours > 0:
        total_time_str += f"{hours}h "
    if minutes > 0:
        total_time_str += f"{minutes}m "
    total_time_str += f"{seconds}s"
    
    print("-" * 80)
    print(f"Total time: {total_time_str}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nResults saved in:")
    print("  - results/exp_6_1_logistic_regression/")
    print("  - results/exp_6_2_mlp/")
    print("  - results/exp_6_3_cnn/")
    print("  - results/exp_6_4_advanced/")
    
    print("\n" + "=" * 80)
    print("Thank you for replicating the Adam paper experiments!")
    print("=" * 80)


if __name__ == '__main__':
    main()

