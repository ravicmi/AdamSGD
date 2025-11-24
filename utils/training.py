"""
Training utilities for model training and validation.
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device, epoch=0, log_interval=100):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on (cpu/cuda/mps)
        epoch (int): Current epoch number for logging
        log_interval (int): Log every N batches
    
    Returns:
        dict: Dictionary with training metrics (loss, accuracy)
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Accumulate loss
        total_loss += loss.item() * data.size(0)
        
        # Update progress bar
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate(model, val_loader, criterion, device, desc='Validation'):
    """
    Validate model on validation/test set.
    
    Args:
        model: Neural network model
        val_loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on (cpu/cuda/mps)
        desc (str): Description for progress bar
    
    Returns:
        dict: Dictionary with validation metrics (loss, accuracy)
    """
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=desc)
        for data, target in pbar:
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, device, 
                num_epochs, log_interval=100):
    """
    Train model for multiple epochs with validation.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on (cpu/cuda/mps)
        num_epochs (int): Number of epochs to train
        log_interval (int): Log every N batches
    
    Returns:
        dict: Dictionary with training history
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, 
            epoch=epoch, log_interval=log_interval
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, 
                              desc=f'Epoch {epoch} [Val]')
        
        # Store metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f'  Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.2f}%')
        print(f'  Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.2f}%')
    
    return history

