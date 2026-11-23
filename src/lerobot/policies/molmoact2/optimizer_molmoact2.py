# Copyright 2026 The Allen Institute for Artificial Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The optimizer of MolmoAct2's fine-tuning recipe. Its preset is `lerobot.optim.MolmoAct2AdamWConfig`."""

import torch


class MolmoAct2AdamW(torch.optim.AdamW):
    """AdamW with component clipping and low-memory BF16 update compensation.

    LeRobot's shared trainer clips the full policy as one vector before calling
    ``Optimizer.step``.  Official MolmoAct2 instead clips each optimizer group
    (LLM, ViT, connector, and action expert) independently, and skips the update
    on every rank when a gradient norm is not finite.  Keeping the clip here lets
    the policy match that behavior without changing the shared trainer or any
    other policy.

    Parameter and optimizer-state dtypes follow the Pi0.5-style storage policy:
    the large VLM tensors remain BF16, while the action expert and explicitly
    sensitive tensors remain FP32. A lazy BF16 Kahan-style residual preserves
    VLM updates smaller than one BF16 parameter ULP. This costs one BF16 tensor
    per trainable BF16 parameter, rather than the FP32 parameter and optimizer
    copies required by the official AMP/FSDP recipe. Native AdamW remains the
    fast path for FP32 parameters (including the complete action expert and
    LoRA adapters).

    Args:
        params (`Iterable`):
            Parameters or parameter groups.
        group_grad_clip_norm (`float`):
            Gradient norm clip of each parameter group.
        **kwargs:
            Passed to `torch.optim.AdamW`.
    """

    def __init__(self, params, *, group_grad_clip_norm: float, **kwargs) -> None:
        if group_grad_clip_norm <= 0:
            raise ValueError(f"MolmoAct2 group_grad_clip_norm must be positive, got {group_grad_clip_norm}.")
        super().__init__(params, **kwargs)
        self.group_grad_clip_norm = float(group_grad_clip_norm)

    def _clip_grad_groups(self) -> tuple[torch.Tensor, ...]:
        norms: list[torch.Tensor] = []
        for group in self.param_groups:
            params_with_grad = [param for param in group["params"] if param.grad is not None]
            if not params_with_grad:
                continue
            norms.append(
                torch.nn.utils.clip_grad_norm_(
                    params_with_grad,
                    max_norm=self.group_grad_clip_norm,
                    error_if_nonfinite=False,
                )
            )
        return tuple(norms)

    def _step_native_non_bfloat16(self) -> None:
        """Run PyTorch's AdamW on the non-bf16 parameters."""
        original_group_params: list[list[torch.Tensor]] = []
        try:
            for group in self.param_groups:
                original_params = group["params"]
                original_group_params.append(original_params)
                group["params"] = [param for param in original_params if param.dtype != torch.bfloat16]
            super().step()
        finally:
            for group, original_params in zip(self.param_groups, original_group_params, strict=True):
                group["params"] = original_params

    def _step_compensated_bfloat16(self) -> None:
        """AdamW on the bf16 parameters, carrying the rounding error to the next step in a bf16 residual."""
        for group in self.param_groups:
            bfloat16_group = dict(group)
            bfloat16_group["params"] = [param for param in group["params"] if param.dtype == torch.bfloat16]

            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            max_exp_avg_sqs: list[torch.Tensor] = []
            state_steps: list[torch.Tensor] = []
            self._init_group(
                bfloat16_group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            beta1, beta2 = group["betas"]
            lr = group["lr"]
            if torch.is_tensor(lr):
                lr = lr.item()
            lr = float(lr)

            for index, parameter in enumerate(params_with_grad):
                grad = grads[index]
                if group["maximize"]:
                    grad = -grad

                state = self.state[parameter]
                compensation = state.get("compensation")
                if compensation is None:
                    compensation = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["compensation"] = compensation

                state_step = state_steps[index]
                state_step.add_(1)
                step_value = state_step.item()

                exp_avg = exp_avgs[index]
                exp_avg_sq = exp_avg_sqs[index]
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1**step_value
                bias_correction2_sqrt = (1 - beta2**step_value) ** 0.5
                step_size = lr / bias_correction1
                if group["amsgrad"]:
                    max_exp_avg_sq = max_exp_avg_sqs[index]
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group["eps"])

                effective_parameter = parameter.float().add_(compensation)
                if group["weight_decay"] != 0:
                    effective_parameter.mul_(1 - lr * group["weight_decay"])
                effective_parameter.addcdiv_(exp_avg, denom, value=-step_size)
                parameter.copy_(effective_parameter)
                compensation.copy_(effective_parameter.sub_(parameter))

    @staticmethod
    def _any_nonfinite_across_ranks(norms: tuple[torch.Tensor, ...]) -> bool:
        """True when any rank has a non-finite gradient norm."""
        local_nonfinite = any(not bool(torch.isfinite(norm).all().item()) for norm in norms)
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_nonfinite

        backend = str(torch.distributed.get_backend()).lower()
        if "nccl" in backend:
            flag_device = torch.device("cuda", torch.cuda.current_device())
        else:
            flag_device = torch.device("cpu")
        nonfinite_flag = torch.tensor(int(local_nonfinite), device=flag_device, dtype=torch.int32)
        torch.distributed.all_reduce(nonfinite_flag, op=torch.distributed.ReduceOp.MAX)
        return bool(nonfinite_flag.item())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        grad_norms = self._clip_grad_groups()
        if self._any_nonfinite_across_ranks(grad_norms):
            # Skip the update on every rank: Adam moments and step counts must not advance.
            self.zero_grad(set_to_none=True)
            return loss
        self._step_native_non_bfloat16()
        self._step_compensated_bfloat16()
        return loss
