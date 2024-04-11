import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self, model, update_after_step=0, inv_gamma=1.0, power=2 / 3, min_value=0.0, max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.alpha = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.alpha = self.get_decay(self.optimization_step)

        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules(), strict=True):
            # Iterate over immediate parameters only.
            for param, ema_param in zip(
                module.parameters(recurse=False), ema_module.parameters(recurse=False), strict=True
            ):
                if isinstance(param, dict):
                    raise RuntimeError("Dict parameter not supported")
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    # Copy BatchNorm parameters, and non-trainable parameters directly.
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.alpha)

        self.optimization_step += 1
