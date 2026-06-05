"""``vllm`` policy: proxy action inference to a remote vLLM OpenPI policy server
(e.g. vllm-omni's GR00T-N1.7 deployment).

This subpackage lives alongside the built-in ``act/``, ``groot/``, ``pi0/``, etc., as an
in-tree lerobot policy. The draccus type ``"vllm"`` is registered when this package is
imported, which happens via ``lerobot.policies.__init__`` (like every other built-in
policy) and through the ``vllm`` branches in ``lerobot.policies.factory``.
"""

from .configuration_vllm import VllmConfig
from .modeling_vllm import VllmPolicy
from .processor_vllm import make_vllm_pre_post_processors

__all__ = ["VllmConfig", "VllmPolicy", "make_vllm_pre_post_processors"]
