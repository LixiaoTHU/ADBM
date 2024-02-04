'''
PGD: Projected Gradient Descent
'''

import torch
from torch import nn, Tensor
from typing import Callable, List
from torch.distributions import laplace, uniform
import numpy as np
from .utils import *
from .AdversarialInputBase import AdversarialInputAttacker
from .rutil import set_random_states, get_random_states
from .L1aux import normalize_by_pnorm




class PGD(AdversarialInputAttacker):
    def __init__(self,
                 model: List[nn.Module],
                 total_step: int = 10,
                 random_start: bool = True,
                 step_size: float = 16 / 255 / 10,
                 eot_step: int = 1,
                 eot_batch_size: int = 1024,
                 criterion: Callable = nn.CrossEntropyLoss(),
                 targeted_attack=False,
                 set_state=False,
                 *args,
                 **kwargs,
                 ):
        self.random_start = random_start
        self.total_step = total_step
        self.step_size = step_size
        self.eot_step = eot_step
        self.eot_batch_size = eot_batch_size
        self.criterion = criterion
        self.targerted_attack = targeted_attack
        self.set_state = set_state
        super(PGD, self).__init__(model, *args, **kwargs)


    def perturb(self, x):
        if self.norm == 'Linf' or self.norm == 'L2':
            x = x + (torch.rand_like(x) - 0.5) * 2 * self.epsilon
            x = clamp(x)
        elif self.norm == "L1":

            ini = laplace.Laplace(
                loc=x.new_tensor(0), scale=x.new_tensor(1)
            )
            delta = ini.sample(x.shape).to(x.device)

            delta = normalize_by_pnorm(delta, p=1)
            ray = uniform.Uniform(0, self.epsilon).sample()
            delta *= ray
            x = clamp(x + delta)

        return x


    def attack(self, x: Tensor, y: Tensor, l1_sparsity=0.95):
        assert len(x.shape) == 4, 'input should have size B, C, H, D'
        original_x = x.clone()
        if self.random_start:
            x = self.perturb(x)

        state = get_random_states()
        for _ in range(self.total_step):
            x.requires_grad = True
            eot_xs = x.repeat(self.eot_step, 1, 1, 1).split(self.eot_batch_size, dim=0)
            eot_ys = y.repeat(self.eot_step).split(self.eot_batch_size, dim=0)
            for i in range(len(eot_xs)):
                eot_x = eot_xs[i]
                eot_y = eot_ys[i]
                logit = 0
                for model in self.models:
                    logit += model(eot_x)
                loss = self.criterion(logit, eot_y)
                loss.backward()
            grad = x.grad / self.eot_step
            x.requires_grad = False
            # update
            if self.norm == 'Linf' or self.norm == 'L2':
                if self.targerted_attack:
                    x -= self.step_size * grad.sign()
                else:
                    x += self.step_size * grad.sign()

            elif self.norm == "L1":
                abs_grad = torch.abs(grad)

                batch_size = grad.size(0)
                view = abs_grad.view(batch_size, -1)
                view_size = view.size(1)

                # vals, idx = view.topk(1)
                vals, idx = view.topk(
                    int(np.round((1 - l1_sparsity) * view_size)))

                out = torch.zeros_like(view).scatter_(1, idx, vals)
                out = out.view_as(grad)
                grad = grad.sign() * (out > 0).float()
                grad = normalize_by_pnorm(grad, p=1)

                if self.targerted_attack:
                    x -= self.step_size * grad
                else:
                    x += self.step_size * grad

                
                
            x = self.clamp(x, original_x)
            if self.set_state:
                set_random_states(state)
        return x, state
