"""
Adam optimizer implementation following Algorithm 1 from the paper:
"Adam: A Method for Stochastic Optimization" by Kingma & Ba (2014)
"""

import torch
from torch.optim.optimizer import Optimizer
import math


class Adam(Optimizer):
    """
    Implements Adam algorithm.
    
    Algorithm 1 from paper:
    m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
    v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate (α in paper). Default: 0.001
        betas (tuple): coefficients for computing running averages (β₁, β₂). Default: (0.9, 0.999)
        eps (float): term added for numerical stability (ε). Default: 1e-8
        weight_decay (float): weight decay (L2 penalty). Default: 0
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)
    
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
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m_t)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (v_t)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Decay the first and second moment running average coefficient
                # m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute step size
                step_size = group['lr'] / bias_correction1
                
                # Compute bias-corrected second moment estimate denominator
                # √v̂_t + ε
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Update parameters
                # θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss

