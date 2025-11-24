"""
SGD with Momentum optimizer implementation.
"""

import torch
from torch.optim.optimizer import Optimizer


class SGDMomentum(Optimizer):
    """
    Implements Stochastic Gradient Descent with Momentum.
    
    Args:
        params: iterable of parameters to optimize
        lr (float): learning rate
        momentum (float): momentum factor. Default: 0.9
        weight_decay (float): weight decay (L2 penalty). Default: 0
        nesterov (bool): whether to use Nesterov momentum. Default: False
    """
    
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super(SGDMomentum, self).__init__(params, defaults)
    
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
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(grad)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad)
                    
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf
                
                p.data.add_(grad, alpha=-group['lr'])
        
        return loss

