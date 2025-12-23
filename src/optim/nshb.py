import torch
from torch.optim import Optimizer

class NSHB(Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                self.state[p] = {
                    "momentum_buffer": torch.zeros_like(p.data)
                }

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                buf = self.state[p]["momentum_buffer"]

                buf.mul_(beta).add_(p.grad, alpha=(1.0 - beta))
                p.add_(buf, alpha=-lr)
