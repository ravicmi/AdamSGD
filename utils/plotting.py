"""
Plotting utilities for visualizing experiment results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(history, optimizer_name, save_path=None, config=None):
    """
    Plot training and validation curves for a single optimizer.
    
    Args:
        history (dict): Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        optimizer_name (str): Name of the optimizer
        save_path (str, optional): Path to save the plot
        config (dict, optional): Configuration for plot settings
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{optimizer_name} - Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{optimizer_name} - Training and Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dpi = config['plotting']['dpi'] if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close()


def plot_comparison(all_histories, metric='train_loss', save_path=None, config=None, 
                   title=None, ylabel=None, use_log_scale=False):
    """
    Plot comparison of multiple optimizers.
    
    Args:
        all_histories (dict): Dictionary mapping optimizer names to their histories
        metric (str): Metric to plot ('train_loss', 'val_loss', 'train_acc', 'val_acc')
        save_path (str, optional): Path to save the plot
        config (dict, optional): Configuration for plot settings
        title (str, optional): Plot title
        ylabel (str, optional): Y-axis label
        use_log_scale (bool): Whether to use log scale for y-axis
    """
    # Get colors from config if available
    if config and 'plotting' in config and 'colors' in config['plotting']:
        colors = config['plotting']['colors']
    else:
        colors = {
            'adam': '#1f77b4',
            'sgd': '#ff7f0e',
            'sgd_momentum': '#2ca02c',
            'adagrad': '#d62728',
            'rmsprop': '#9467bd'
        }
    
    plt.figure(figsize=(10, 6))
    
    for optimizer_name, history in all_histories.items():
        if metric in history and len(history[metric]) > 0:
            epochs = range(1, len(history[metric]) + 1)
            color = colors.get(optimizer_name.lower(), None)
            label = optimizer_name.replace('_', ' ').title()
            plt.plot(epochs, history[metric], label=label, linewidth=2.5, 
                    color=color, marker='o', markersize=4, markevery=max(1, len(epochs)//10))
    
    plt.xlabel('Epoch', fontsize=14)
    
    if ylabel:
        plt.ylabel(ylabel, fontsize=14)
    else:
        if 'loss' in metric:
            plt.ylabel('Loss', fontsize=14)
        elif 'acc' in metric:
            plt.ylabel('Accuracy (%)', fontsize=14)
    
    if title:
        plt.title(title, fontsize=16, fontweight='bold')
    else:
        metric_name = metric.replace('_', ' ').title()
        plt.title(f'Optimizer Comparison - {metric_name}', fontsize=16, fontweight='bold')
    
    if use_log_scale:
        plt.yscale('log')
    
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dpi = config['plotting']['dpi'] if config else 300
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.close()


def plot_multiple_comparisons(all_histories, save_dir, config=None, experiment_name=''):
    """
    Generate all comparison plots for an experiment.
    
    Args:
        all_histories (dict): Dictionary mapping optimizer names to their histories
        save_dir (str): Directory to save plots
        config (dict, optional): Configuration for plot settings
        experiment_name (str): Name of the experiment for plot titles
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loss comparison
    plot_comparison(
        all_histories, 
        metric='train_loss',
        save_path=os.path.join(save_dir, 'comparison_train_loss.png'),
        config=config,
        title=f'{experiment_name} - Training Loss Comparison',
        ylabel='Training Loss'
    )
    
    # Validation loss comparison
    plot_comparison(
        all_histories,
        metric='val_loss',
        save_path=os.path.join(save_dir, 'comparison_val_loss.png'),
        config=config,
        title=f'{experiment_name} - Validation Loss Comparison',
        ylabel='Validation Loss'
    )
    
    # Training accuracy comparison
    plot_comparison(
        all_histories,
        metric='train_acc',
        save_path=os.path.join(save_dir, 'comparison_train_acc.png'),
        config=config,
        title=f'{experiment_name} - Training Accuracy Comparison',
        ylabel='Training Accuracy (%)'
    )
    
    # Validation accuracy comparison
    plot_comparison(
        all_histories,
        metric='val_acc',
        save_path=os.path.join(save_dir, 'comparison_val_acc.png'),
        config=config,
        title=f'{experiment_name} - Validation Accuracy Comparison',
        ylabel='Validation Accuracy (%)'
    )
    
    print(f"All comparison plots saved to: {save_dir}")

