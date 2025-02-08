import abc
import math
from dataclasses import asdict, dataclass

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LRScheduler


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0"""

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        del num_training_steps

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, self.num_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.num_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < self.num_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_annealing")
@dataclass
class CosineAnnealingSchedulerConfig(LRSchedulerConfig):
    """Implements Cosine Annealing learning rate scheduler"""

    min_lr: float = 0  # Minimum learning rate
    T_max: int = 100000  # Number of iterations for a full decay (half-cycle)
    num_warmup_steps: int = 0  # Not used but somehow required by the parent class

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler:
        return CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.min_lr)
