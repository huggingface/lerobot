# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import abc
import builtins
import dataclasses
import functools
import itertools
import logging
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict, TypeVar, Unpack

import torch
from huggingface_hub import hf_hub_download, save_torch_state_dict
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from torch import Tensor, nn

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import resolve_safetensors_device
from lerobot.utils.hub import HubMixin
from lerobot.utils.import_utils import _peft_available, require_package

from .utils import log_model_loading_keys

if TYPE_CHECKING or _peft_available:
    from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType, get_peft_model
else:
    PEFT_TYPE_TO_CONFIG_MAPPING = None
    PeftType = None
    get_peft_model = None

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

T = TypeVar("T", bound="PreTrainedPolicy")

# Pinned far above any policy's total size so save_torch_state_dict always emits exactly one
# `model.safetensors` (no shards, no index) — a constant, not a computed byte count.
_SINGLE_FILE_SHARD_SIZE = "1TB"


@functools.cache
def _compile_fp32_paths(paths: tuple[str, ...]) -> re.Pattern[str] | None:
    """Compile `_fp32_modules` dotted paths into one alternation over full tensor names.

    Matching is anchored at dot-segment boundaries but not at the start of the name, so
    `"model.norm"` matches `model.norm.weight` and `base_model.model.model.norm.weight` alike,
    while `"norm"` does not match `post_attention_layernorm.weight`. `*` is exactly one segment.
    """
    if not paths:
        return None
    branches = []
    for path in paths:
        segments = path.split(".")
        if not all(segments) or any("*" in s and s != "*" for s in segments):
            raise ValueError(
                f"Invalid path {path!r} in _fp32_modules: use dot-separated names, with '*' as a "
                "whole segment to match one level."
            )
        body = r"\.".join(r"[^.]+" if s == "*" else re.escape(s) for s in segments)
        branches.append(rf"(?:(?:^|\.){body}(?:\.|$))")
    return re.compile("|".join(branches))


def _wrap_init_with_post_init(cls: builtins.type) -> None:
    """Make `post_init()` run once, after the outermost `__init__` of a policy returns.

    `PreTrainedPolicy.__init__` runs before a subclass has built anything, so the base class cannot
    finalize precision there. transformers solves this by asking every subclass to call
    `self.post_init()` at the end of its `__init__`; a policy that forgets gets a `config.dtype`
    its tensors do not honour, silently. Wrapping instead makes the contract an invariant of the
    base class, including for third-party policies, at no cost to the author.

    The depth counter makes a policy that subclasses another policy finalize exactly once, after
    the most-derived `__init__` returns. Exceptions propagate unchanged.
    """
    # Fall back to the inherited `__init__` so a subclass that defines none is still covered; if it
    # was inherited from an already-wrapped policy the marker short-circuits here.
    original = cls.__dict__.get("__init__") or cls.__init__
    if getattr(original, "_lerobot_runs_post_init", False):
        return

    @functools.wraps(original)
    def init_then_post_init(self, *args: Any, **kwargs: Any) -> None:
        depth = getattr(self, "_lerobot_init_depth", 0)
        object.__setattr__(self, "_lerobot_init_depth", depth + 1)
        try:
            original(self, *args, **kwargs)
        finally:
            object.__setattr__(self, "_lerobot_init_depth", depth)
        if depth == 0:
            self.post_init()
            # From here on, a conversion means somebody is changing a built policy's precision,
            # which desynchronizes anything already derived from its parameters.
            object.__setattr__(self, "_lerobot_construction_finished", True)

    init_then_post_init._lerobot_runs_post_init = True  # type: ignore[attr-defined]
    cls.__init__ = init_then_post_init


class ActionSelectKwargs(TypedDict, total=False):
    noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    # --- declarative precision surface -------------------------------------------------------
    # Tensors that must stay in float32 however `config.dtype` is set, because the policy is
    # numerically unstable without them: action heads, flow-matching time embeddings,
    # normalization statistics, rotary caches.
    #
    # An entry is either a dotted path, matched against full parameter/buffer names at dot-segment
    # boundaries and at any depth (`*` stands for exactly one segment), or an `nn.Module` subclass,
    # which protects every floating tensor of every module of that type. Best practice is a
    # complete path from the policy root, which reads directly against `named_parameters()`;
    # matching at any depth means a wrapper inserted above the root (PEFT's `base_model.model.`,
    # `torch.compile`'s `_orig_mod.`, DDP's `module.`) does not silently void the rule.
    #
    # Declare these here on the policy class, not on nested modules, so a reader finds the whole
    # precision layout in one place. If neither form can express what a policy needs, override
    # `_fp32_tensor_names()`.
    _fp32_modules: ClassVar[tuple[str | type[nn.Module], ...]] = ()

    # --- declarative parallelism/acceleration surface ----------------------------------------
    # Module CLASS names forming the FSDP2 wrap units (and, once wired, the activation-
    # checkpointing units). Resolved onto the accelerate plugin right before
    # `accelerator.prepare()` by `lerobot.distributed.set_fsdp_wrap_modules`; sharded training
    # with no wrap source anywhere fails loudly instead of silently wrapping only the root.
    _fsdp_wrap_modules: ClassVar[list[str] | None] = None
    # Non-`forward` entry points that must trigger FSDP2 unshard/reshard hooks when called on a
    # sharded policy (registered post-prepare via `torch.distributed.fsdp
    # .register_fsdp_forward_method`); calling them unregistered crashes on mixed Tensor/DTensor.
    _fsdp_forward_methods: ClassVar[tuple[str, ...]] = ("select_action", "predict_action_chunk")
    # Capability gate for the (future) activation-checkpointing wiring.
    supports_gradient_checkpointing: ClassVar[bool] = False
    # Declarative context-parallel plan (diffusers `ContextParallelModelPlan` semantics:
    # module FQN -> sequence split/gather spec). Reserved for the CP engine round.
    _cp_plan: ClassVar[dict[str, Any] | None] = None
    # Attribute names `drop_queued_actions` clears: a `populate_queues`-style `_queues` dict
    # and a bare `_action_queue` deque. A chunking policy using another name must extend this
    # ClassVar (or override the method), otherwise dropping the queue silently does nothing.
    _action_queue_attrs: ClassVar[tuple[str, ...]] = ("_queues", "_action_queue")

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    # --- parameter precision -----------------------------------------------------------------

    def _fp32_tensor_names(self) -> set[str]:
        """Full names of the floating tensors `_fp32_modules` protects, resolved against `self`.

        Computed from the live registrations on every call, so it cannot go stale when modules or
        parameter ties change. If any alias of a shared tensor matches, every alias is protected.
        Override this to express a layout the declarative forms cannot.
        """
        if not self._fp32_modules:
            return set()

        names: set[str] = set()
        module_types = tuple(entry for entry in self._fp32_modules if isinstance(entry, type))
        if module_types:
            for module_name, module in self.named_modules():
                if isinstance(module, module_types):
                    prefix = f"{module_name}." if module_name else ""
                    for name, _ in itertools.chain(module.named_parameters(), module.named_buffers()):
                        names.add(prefix + name)

        pattern = _compile_fp32_paths(tuple(e for e in self._fp32_modules if isinstance(e, str)))
        registrations = list(
            itertools.chain(
                self.named_parameters(remove_duplicate=False), self.named_buffers(remove_duplicate=False)
            )
        )
        if pattern is not None:
            names.update(name for name, _ in registrations if pattern.search(name))

        # Propagate to every alias of a shared tensor, so protecting one name protects the object.
        aliases: dict[int, list[str]] = defaultdict(list)
        for name, tensor in registrations:
            aliases[id(tensor)].append(name)
        for tensor_names in aliases.values():
            if not names.isdisjoint(tensor_names):
                names.update(tensor_names)
        return names

    def post_init(self) -> None:
        """Bring the constructed policy onto the precision layout `config.dtype` asks for.

        Runs automatically once the outermost `__init__` returns, so a policy author never has to
        remember it. Idempotent and re-runnable: an explicit call is allowed and harmless.

        A policy that already builds its submodules at the requested precision — by forwarding
        `config.dtype` into whatever constructs them — leaves this with nothing to do. It is a
        guarantee and a fallback, not the primary mechanism.
        """
        requested = self.config.dtype
        if requested is None:
            # "Unspecified" means the policy is left exactly as its __init__ built it.
            return
        protected = self._fp32_tensor_names()
        converted: list[str] = []
        views: list[str] = []
        with torch.no_grad():
            for name, tensor in itertools.chain(self.named_parameters(), self.named_buffers()):
                if not tensor.is_floating_point():
                    continue
                target = torch.float32 if name in protected else requested
                if tensor.dtype is target:
                    continue
                if tensor._base is not None:
                    # A tensor that is a view of another one gets its own storage here, so it stops
                    # tracking its base. Object-identity ties are fine (`.data =` mutates the shared
                    # object in place); storage-only sharing is not, so say so rather than silently
                    # decoupling them.
                    views.append(name)
                converted.append(name)
                # Assigning `.data` keeps the nn.Parameter object, and therefore every alias of a
                # shared tensor, intact. Rewrapping in a fresh Parameter would break ties.
                tensor.data = tensor.data.to(target)
        if views:
            warnings.warn(
                f"{type(self).__name__}.post_init() converted {', '.join(views)}, which are views of "
                "other tensors; they now own their storage and no longer track their base. Register "
                "them as independent tensors, or add them to _fp32_modules alongside their base.",
                UserWarning,
                stacklevel=2,
            )
        if converted and getattr(self, "_lerobot_construction_finished", False):
            warnings.warn(
                f"{type(self).__name__}.post_init() changed parameter dtypes after construction had "
                "finished. Any optimizer, compiled graph or distributed wrapper built from this "
                "policy now disagrees with its parameters. Precision is meant to be chosen through "
                "config.dtype before the policy is built.",
                UserWarning,
                stacklevel=2,
            )
        if converted:
            logging.debug(
                "%s.post_init() converted %d tensor(s) that __init__ left at another precision: %s",
                type(self).__name__,
                len(converted),
                ", ".join(converted[:10]) + (" ..." if len(converted) > 10 else ""),
            )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype holding most of this policy's floating parameter memory.

        Reports what the tensors are, not what `config.dtype` asked for, and is never written back
        into the config. Weighted by `numel()` so a handful of protected float32 tensors cannot
        misreport a bfloat16 policy — the failure mode of the "first floating parameter" shortcut.
        Buffers are excluded: they are derived state, and a large float32 statistics buffer should
        not outvote the weights.

        Iterates the parameters on each call (single-digit milliseconds for a multi-billion
        parameter policy), so treat it as a diagnostic, not a per-step accessor. Use
        `parameter_dtypes()` to see the full breakdown.
        """
        counts = self.parameter_dtypes()
        if not counts:
            return self.config.dtype or torch.get_default_dtype()
        most = max(counts.values())
        candidates = [dtype for dtype, count in counts.items() if count == most]
        # On a tie, report what was asked for rather than whichever came first.
        if self.config.dtype in candidates:
            return self.config.dtype
        return candidates[0]

    def parameter_dtypes(self) -> dict[torch.dtype, int]:
        """Number of floating parameter elements held in each dtype."""
        counts: dict[torch.dtype, int] = defaultdict(int)
        for parameter in self.parameters():
            if parameter.is_floating_point():
                counts[parameter.dtype] += parameter.numel()
        return dict(counts)

    def _apply(self, *args, **kwargs):
        """Warn when a post-construction cast would erase the declared float32 exceptions.

        Behaviour is unchanged — `.to()`, `.half()`, `.float()`, `.bfloat16()` all do exactly what
        PyTorch does. But casting a policy after it is built is not a supported way to choose its
        precision, and it silently drops the protections the policy declared, so say so. Hooking
        `_apply` covers every public cast API with one override; the probe is O(1), so the
        `.to(device)` that every run performs costs two `next()` calls and never warns.
        """
        if not self._fp32_modules:
            return super()._apply(*args, **kwargs)
        before = next((p.dtype for p in self.parameters() if p.is_floating_point()), None)
        result = super()._apply(*args, **kwargs)
        after = next((p.dtype for p in self.parameters() if p.is_floating_point()), None)
        if before is not None and after is not None and before is not after:
            declared = ", ".join(e if isinstance(e, str) else e.__name__ for e in self._fp32_modules)
            warnings.warn(
                f"Casting {type(self).__name__} after construction also casts the tensors it keeps "
                f"in float32 ({declared}), which can change its numerics. Set config.dtype and "
                f"rebuild through make_policy to get a supported precision layout.",
                UserWarning,
                stacklevel=3,
            )
        return result

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")
        _wrap_init_with_post_init(cls)
        # The rollout stack gates text queries on supports_text_generation(), so a
        # generate_text() override without it is unreachable. Compared through the MRO, so an
        # override inherited from a conforming parent counts.
        if (
            cls.generate_text is not PreTrainedPolicy.generate_text
            and cls.supports_text_generation is PreTrainedPolicy.supports_text_generation
        ):
            raise TypeError(
                f"{cls.__name__} provides generate_text() but not supports_text_generation(). "
                "Override supports_text_generation() too (returning True, or a "
                "checkpoint-conditional value), otherwise the text head is never used."
            )

    def _save_pretrained(self, save_directory: Path) -> None:
        """Serialize this policy's parameters (and config) into `save_directory`.

        Sharding is handled internally: under FSDP2 the full state dict is gathered through a
        COLLECTIVE, so when the policy is sharded this method (via `save_pretrained`) must be
        called on EVERY rank — a rank-0-gated call deadlocks. File writes happen on the main
        process only, in all layouts (single, DDP, sharded).

        Args:
            save_directory (Path): Target directory for the policy config (`config.json`) and the
                safetensors weight file(s).
        """
        # Lazy imports: the persistence layer pulls in lerobot.distributed only when saving.
        from lerobot.distributed.checkpoint import full_model_state_dict, is_sharded_module
        from lerobot.distributed.utils import is_main_process

        model_to_save = self.module if hasattr(self, "module") else self
        if is_sharded_module(model_to_save):
            logging.info("Gathering the full state dict from all ranks (sharded policy).")
        state_dict = full_model_state_dict(model_to_save)  # collective when sharded; {} off-main
        if not state_dict or not is_main_process():
            # Sharded: the gather materializes on the main rank only (emptiness check).
            # Non-sharded multi-rank (DDP): every rank holds a full dict — the explicit rank
            # gate prevents N ranks racing on the same files. Single process: never taken.
            return
        # `config.dtype` is the request that reproduces this policy, so it is saved as-is. If the
        # tensors no longer match it, somebody cast the policy after construction: say so rather
        # than rewriting their request. Reloading rebuilds at config.dtype and casts these weights
        # into it, which is lossless in the usual (low precision -> float32) direction.
        if self.config.dtype is not None and (observed := model_to_save.dtype) is not self.config.dtype:
            logging.warning(
                "Saving %s whose parameters are mostly %s while config.dtype requests %s (breakdown: %s). "
                "The config records the request, so reloading rebuilds the policy at %s.",
                type(model_to_save).__name__,
                observed,
                self.config.dtype,
                model_to_save.parameter_dtypes(),
                self.config.dtype,
            )
        self.config._save_pretrained(save_directory)
        save_torch_state_dict(state_dict, str(save_directory), max_shard_size=_SINGLE_FILE_SHARD_SIZE)

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        missing_keys, unexpected_keys = load_model_as_safetensor(
            model, model_file, strict=strict, device=resolve_safetensors_device(map_location)
        )
        log_model_loading_keys(missing_keys, unexpected_keys)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    def drop_queued_actions(self) -> None:
        """Discard actions precomputed by earlier ``select_action`` calls.

        Forces a fresh forward pass on the next ``select_action`` so a mid-episode conditioning
        change (e.g. a new instruction) takes effect at once instead of after the queue drains.
        Unlike :meth:`reset`, the rest of the episode state is kept.  Call it from the thread
        that calls ``select_action``.  Clears the queues named in :attr:`_action_queue_attrs`;
        a policy that keeps no action queue inherits a no-op.
        """
        for attr in self._action_queue_attrs:
            queue = getattr(self, attr, None)
            if isinstance(queue, dict):
                # populate_queues-style dict: clear only ACTION, not the observation history.
                if ACTION in queue:
                    queue[ACTION].clear()
            elif queue is not None:
                queue.clear()

    def supports_rtc(self) -> bool:
        """Whether this policy implements Real-Time Chunking inference semantics."""
        return False

    def supports_text_generation(self) -> bool:
        """Whether this policy implements :meth:`generate_text` (override both together)."""
        return False

    def generate_text(self, batch: dict[str, Any]) -> str:
        """Run the policy's text head on a preprocessed observation batch.

        The request rides on ``batch`` as complementary data (:data:`~lerobot.utils.constants.QUERY_KIND`
        / ``QUERY_TEXT``); a ``next_subtask`` reply is fed straight into ``set_task``, so it must be
        exactly one subtask, not a plan or a numbered list.  Returns the generated text, and must not
        mutate action-producing state (queues, observation history).
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no text head — it cannot answer questions or plan subtasks."
        )

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Returns the action chunk (for action chunking policies) for a given observation, potentially in batch mode.

        Child classes using action chunking should use this method within `select_action` to form the action chunk
        cached for selection.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
        peft_model=None,
        state_dict: dict[str, Tensor] | None = None,
        dataset_meta: LeRobotDatasetMetadata | None = None,
    ) -> None:
        """Publish this policy to the Hub.

        Deprecated: use :func:`lerobot.common.train_utils.publish_trained_model` instead, which
        also publishes the pre/post-processors alongside the model.

        Args:
            cfg (TrainPipelineConfig): The training config; saved as `train_config.json` and
                used to render the model card.
            peft_model: The PEFT wrapper when training adapters, whose weights replace the full
                model weights in the published repo. Defaults to None.
            state_dict (dict[str, Tensor] | None): Ignored; weights are now gathered internally
                when the policy is sharded. Defaults to None.
            dataset_meta (LeRobotDatasetMetadata | None): Dataset metadata for the model card,
                if available. Defaults to None.
        """
        from lerobot.common.train_utils import publish_trained_model

        warnings.warn(
            "PreTrainedPolicy.push_model_to_hub is deprecated and will be removed in a future "
            "version. Use lerobot.common.train_utils.publish_trained_model(cfg, model, "
            "preprocessor, postprocessor, dataset_meta) instead.",
            FutureWarning,
            stacklevel=2,
        )
        if state_dict is not None:
            warnings.warn(
                "The `state_dict` argument is ignored: sharded weights are gathered internally "
                "when the policy is saved.",
                FutureWarning,
                stacklevel=2,
            )
        publish_trained_model(cfg, self, None, None, dataset_meta, peft_model=peft_model)

    def wrap_with_peft(
        self,
        peft_config=None,
        peft_cli_overrides: dict | None = None,
    ) -> PreTrainedPolicy:
        """
        Wrap this policy with PEFT adapters for parameter-efficient fine-tuning.

        This method is the single entry point for PEFT integration. Subclasses should
        override `_get_default_peft_targets()` to provide default target modules, and
        `_validate_peft_config()` for policy-specific validation.

        Args:
            peft_config: Optional PEFT adapter configuration (e.g., LoraConfig).
                If provided, used directly (with CLI overrides applied).
            peft_cli_overrides: Optional dict of CLI overrides (method_type, target_modules, r, etc.)
                These are merged with policy defaults to build the final config.
        """
        require_package("peft", extra="peft")

        # If user provided a complete config, use it directly (with overrides)
        if peft_config is not None:
            final_config = peft_config
            if peft_cli_overrides:
                final_config = self._apply_peft_cli_overrides(final_config, peft_cli_overrides)
        else:
            # Build config from defaults + CLI overrides
            final_config = self._build_peft_config(peft_cli_overrides or {})

        # Validate the configuration
        self._validate_peft_config(final_config)

        # Freeze base parameters, only adapter params will be trained
        for p in self.parameters():
            p.requires_grad_(False)

        # Store pretrained path for PEFT's base_model_name_or_path
        if self.config.pretrained_path:
            self.name_or_path = str(self.config.pretrained_path)

        # Wrap with PEFT
        peft_model = get_peft_model(self, final_config)

        # Mark config as using PEFT for proper loading later
        peft_model.config.use_peft = True

        logging.info(f"Wrapped {self.name} with PEFT ({type(final_config).__name__})")
        return peft_model

    def _get_default_peft_targets(self) -> dict[str, any] | None:
        """
        Return default PEFT target modules for this policy.

        Override this in subclasses to provide policy-specific defaults. These defaults
        are PEFT-method agnostic - they only specify which modules to target.

        """
        return None

    def _validate_peft_config(self, peft_config) -> None:
        """
        Validate the PEFT configuration for this policy.

        Override this in subclasses to add policy-specific validation or warnings.
        The default implementation checks that a pretrained_path exists.

        Args:
            peft_config: The PEFT configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.config.pretrained_path:
            raise ValueError(
                "Training from scratch using PEFT is unlikely to yield good results. "
                "Supply a `policy.pretrained_path` to fine-tune an existing model."
            )

    def _preprocess_peft_cli_overrides(self, cli_overrides: dict, peft_method_type) -> dict:
        """
        Preprocess CLI overrides: rename keys and handle method-specific init_type.

        Args:
            cli_overrides: Dict of CLI options (will be copied, not mutated).
            peft_method_type: The PeftType enum value for the PEFT method.

        Returns:
            Preprocessed dict with renamed keys and init_type mapped to method-specific key.
        """
        require_package("peft", extra="peft")

        cli_overrides = cli_overrides.copy()

        # Handle the full_training_modules -> modules_to_save rename
        if "full_training_modules" in cli_overrides:
            cli_overrides["modules_to_save"] = cli_overrides.pop("full_training_modules")

        # Remove method_type as it's handled separately
        cli_overrides.pop("method_type", None)

        # Handle init_type specially based on PEFT method
        init_type = cli_overrides.pop("init_type", None)
        if init_type is not None:
            if peft_method_type == PeftType.LORA:
                cli_overrides["init_lora_weights"] = init_type
            elif peft_method_type == PeftType.MISS:
                cli_overrides["init_weights"] = init_type
            else:
                raise ValueError(f"Init type '{init_type}' unknown for PEFT method {peft_method_type}.")

        return cli_overrides

    def _build_peft_config(self, cli_overrides: dict):
        """Build a PEFT config from policy defaults and CLI overrides."""
        require_package("peft", extra="peft")

        # Determine PEFT method type (default to LORA)
        method_type_str = cli_overrides.get("method_type") or "lora"
        peft_method_type = PeftType[method_type_str.upper()]
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

        # Preprocess CLI overrides
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with policy defaults, apply CLI overrides
        config_dict = dict(self._get_default_peft_targets() or {})
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        # Ensure we have target_modules
        if not config_dict.get("target_modules"):
            raise ValueError(
                f"Policy '{self.name}' does not define default target_modules. "
                "Please pass --peft.target_modules explicitly."
            )

        return peft_config_cls(**config_dict)

    def _apply_peft_cli_overrides(self, peft_config, cli_overrides: dict):
        """Apply CLI overrides to an existing PEFT config."""
        require_package("peft", extra="peft")

        # Get method type from existing config or CLI override
        method_type_str = cli_overrides.get("method_type")
        if method_type_str:
            peft_method_type = PeftType[method_type_str.upper()]
            peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]
        else:
            peft_method_type = PeftType(peft_config.peft_type)
            peft_config_cls = type(peft_config)

        # Preprocess CLI overrides
        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        # Start with existing config, apply CLI overrides
        config_dict = {k: v for k, v in dataclasses.asdict(peft_config).items() if not k.startswith("_")}
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        return peft_config_cls(**config_dict)
