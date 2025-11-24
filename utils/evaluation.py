"""
Evaluation utilities for model assessment.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def calculate_accuracy(output, target):
    """
    Calculate classification accuracy.
    
    Args:
        output: Model output logits
        target: Ground truth labels
    
    Returns:
        float: Accuracy percentage
    """
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / target.size(0)
    return accuracy


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Neural network model
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on (cpu/cuda/mps)
    
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Accumulate loss
            total_loss += loss.item() * data.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

