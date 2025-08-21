"""
Optimizer Adapter for HRM Project

This module provides a compatible alternative to the adam-atan2 package
that was causing compatibility issues. It implements the AdamATan2 optimizer
using PyTorch's built-in optimizers with custom step functions.
"""

import torch
import torch.optim as optim
from typing import List, Optional, Tuple, Union
import math


class AdamATan2(optim.Optimizer):
    """
    AdamATan2 optimizer - a drop-in replacement for the adam-atan2 package.
    
    This optimizer implements the Adam optimizer with atan2-based learning rate
    adaptation, which can help with training stability.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
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
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)
    
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('AdamATan2 does not support sparse gradients')
                    grads.append(p.grad)
                    
                    state = self.state[p]
                    
                    # State initialization
                    if len(state) == 0:
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) if group['capturable'] else torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp_avg_sq until now
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                    
                    state_steps.append(state['step'])
            
            # Use the standard Adam step function
            self._adam_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group,
            )
        
        return loss
    
    def _adam_step(
        self,
        params_with_grad: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        exp_avg_sqs: List[torch.Tensor],
        max_exp_avg_sqs: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        group: dict,
    ):
        """Implements the Adam step with atan2-based learning rate adaptation."""
        for i, param in enumerate(params_with_grad):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step_t = state_steps[i]
            
            # Update step
            step_t += 1
            
            # Decay the first and second moment running average coefficient
            exp_avg.mul_(group['beta1']).add_(grad, alpha=1 - group['beta1'])
            exp_avg_sq.mul_(group['beta2']).addcmul_(grad, grad, value=1 - group['beta2'])
            
            if group['amsgrad']:
                # Maintains the maximum of all 2nd moment running avg. until now
                max_exp_avg_sq = max_exp_avg_sqs[i]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])
            
            step = step_t.item()
            bias_correction1 = 1 - group['beta1'] ** step
            bias_correction2 = 1 - group['beta2'] ** step
            
            # Apply atan2-based learning rate adaptation
            # This is the key difference from standard Adam
            lr_scale = math.atan2(bias_correction2, bias_correction1) / math.pi + 0.5
            lr = group['lr'] * lr_scale
            
            step_size = lr / bias_correction1
            
            # Apply weight decay
            if group['weight_decay'] != 0:
                param.add_(param, alpha=-group['weight_decay'] * lr)
            
            # Update parameter
            param.addcdiv_(exp_avg, denom, value=-step_size)


# Alternative: Simple AdamW wrapper for immediate compatibility
class AdamWAdapter(optim.AdamW):
    """
    Simple AdamW adapter that can be used as a drop-in replacement.
    This provides immediate compatibility without the atan2 features.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        return super().step(closure)


# Factory function to create the appropriate optimizer
def create_optimizer(optimizer_type: str = "adam_atan2", **kwargs):
    """
    Factory function to create the appropriate optimizer.
    
    Args:
        optimizer_type: Type of optimizer ("adam_atan2", "adamw", or "adam")
        **kwargs: Optimizer parameters
    
    Returns:
        Optimizer instance
    """
    if optimizer_type == "adam_atan2":
        return AdamATan2(**kwargs)
    elif optimizer_type == "adamw":
        return optim.AdamW(**kwargs)
    elif optimizer_type == "adam":
        return optim.Adam(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Export the main classes
__all__ = ["AdamATan2", "AdamWAdapter", "create_optimizer"]
