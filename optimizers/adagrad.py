"""
AdaGrad optimizer implementation.
Reference: Duchi et al., "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization" (2011)
"""

import torch
from torch.optim.optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    Implements AdaGrad algorithm.
    
    AdaGrad adapts the learning rate to the parameters, performing smaller updates
    for parameters associated with frequently occurring features, and larger updates
    for parameters associated with infrequent features.
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate. Default: 0.01
        lr_decay (float): learning rate decay. Default: 0
        eps (float): term added to the denominator for numerical stability. Default: 1e-8
        weight_decay (float): weight decay (L2 penalty). Default: 0
    """
    
    def __init__(self, params, lr=0.01, lr_decay=0, eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= lr_decay:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay)
        super(AdaGrad, self).__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['sum'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Accumulate gradient squares
                state['sum'].addcmul_(grad, grad, value=1)
                
                # Compute learning rate with decay
                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])
                
                # Update parameters
                std = state['sum'].sqrt().add_(group['eps'])
                p.data.addcdiv_(grad, std, value=-clr)
        
        return loss

