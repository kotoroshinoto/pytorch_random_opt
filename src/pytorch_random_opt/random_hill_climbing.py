import torch
from torch.optim.optimizer import Optimizer
import random
import math


class RandomHillClimbing(Optimizer):
    def __init__(self, params, lr=0.1, num_restarts=1):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_restarts < 1:
            raise ValueError("Invalid number of restarts: {}".format(num_restarts))

        self.num_restarts = num_restarts
        defaults = dict(lr=lr)
        super(RandomHillClimbing, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Random Hill Climbing step
                with torch.no_grad():
                    best_loss = float('inf')
                    for _ in range(self.num_restarts):
                        noise = torch.randn_like(p.data) * lr
                        p.data += noise
                        current_loss = closure()
                        if current_loss < best_loss:
                            best_loss = current_loss
                        else:
                            p.data -= noise  # Revert to previous position

        return loss
