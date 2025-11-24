"""
RMSProp optimizer implementation.
Reference: Tieleman & Hinton, "Lecture 6.5 - RMSProp" (2012)
"""

import torch
from torch.optim.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    Implements RMSProp algorithm.
    
    RMSProp divides the learning rate by an exponentially decaying average of squared gradients.
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate. Default: 0.001
        alpha (float): smoothing constant. Default: 0.99
        eps (float): term added to the denominator for numerical stability. Default: 1e-8
        momentum (float): momentum factor. Default: 0
        weight_decay (float): weight decay (L2 penalty). Default: 0
    """
    
    def __init__(self, params, lr=0.001, alpha=0.99, eps=1e-8, momentum=0, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super(RMSProp, self).__init__(params, defaults)
    
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
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                
                square_avg = state['square_avg']
                alpha = group['alpha']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update biased second moment estimate
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                
                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, square_avg.sqrt().add_(group['eps']))
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])
                    p.data.addcdiv_(grad, avg, value=-group['lr'])
        
        return loss

