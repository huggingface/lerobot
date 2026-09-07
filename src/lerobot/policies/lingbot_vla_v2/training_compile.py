"""Keep checkpointed decoder subtrees eager without removing FSDP wrappers."""

from torch import nn
from torch._dynamo.eval_frame import OptimizedModule


def _unwrap_compiled_subtrees(model: nn.Module, subtree_types: tuple[type[nn.Module], ...]) -> int:
    """Remove only OptimizedModule wrappers inside selected decoder subtrees.

    A compiled FSDP decoder's ``_orig_mod`` is still an FSDPModule: keep that
    exact object, its parameters, and its hooks. This runs after Accelerate
    prepares model/optimizer but before the first forward. Decorating an outer
    function with compiler.disable alone is insufficient: an OptimizedModule
    can enter its own Dynamo context inside that function.
    """
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    count = 0

    def visit(parent: nn.Module, selected: bool = False) -> None:
        nonlocal count
        for name, child in tuple(parent.named_children()):
            original = child._orig_mod if isinstance(child, OptimizedModule) else child
            in_subtree = selected or isinstance(original, subtree_types)
            if in_subtree and isinstance(child, OptimizedModule):
                parent.add_module(name, original)
                count += 1
                child = original
            visit(child, in_subtree)

    visit(model)
    if parameter_ids != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("Removing decoder compile wrappers changed optimizer parameter identities")
    return count


def uncompile_checkpointed_decoders(model: nn.Module) -> int:
    from .model_core.qwen2_action_expert import Qwen2DecoderLayer
    from .model_core.qwen3vl_in_vla import Qwen3VLTextDecoderLayer

    return _unwrap_compiled_subtrees(model, (Qwen3VLTextDecoderLayer, Qwen2DecoderLayer))
