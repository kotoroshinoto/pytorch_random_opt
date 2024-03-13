import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class SimulatedAnnealing(Optimizer):
    def __init__(self, params, lr=0.1, T_initial=1, T_min=0.001, alpha=0.99, decay='exponential'):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if T_initial < 0.0:
            raise ValueError("Invalid initial temperature: {}".format(T_initial))
        if T_min < 0.0:
            raise ValueError("Invalid minimum temperature: {}".format(T_min))
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if decay not in ['arithmetic', 'geometric', 'exponential']:
            raise ValueError("Invalid decay type: {}".format(decay))

        self.decay = decay
        defaults = dict(lr=lr, T_initial=T_initial, T_min=T_min, alpha=alpha)
        super(SimulatedAnnealing, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            T_initial = group['T_initial']
            T_min = group['T_min']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # Simulated Annealing step
                with torch.no_grad():
                    p.data -= lr * d_p
                    # Simulated Annealing update
                    T = self._get_temperature(T_initial, alpha)
                    for idx, val in enumerate(p.data.view(-1)):
                        if np.random.random() < np.exp(-val/T):
                            p.data.view(-1)[idx] = np.random.uniform(-1, 1)

        self.state['iteration'] = self.state.get('iteration', 0) + 1
        return loss

    def _get_temperature(self, T_initial, alpha):
        if self.decay == 'arithmetic':
            return max(T_initial - alpha * self.state.get('iteration', 0), self.defaults['T_min'])
        elif self.decay == 'geometric':
            return max(T_initial * alpha ** self.state.get('iteration', 0), self.defaults['T_min'])
        elif self.decay == 'exponential':  # exponential decay
            return max(T_initial * (alpha ** self.state.get('iteration', 0)), self.defaults['T_min'])
        else:
            raise ValueError("Invalid decay type: {}".format(self.decay))
