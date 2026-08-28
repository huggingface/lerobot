from torch import Tensor
from torch.nn import functional as F  # noqa: N812


def optional_binary_loss(logits: Tensor, labels: Tensor | None) -> Tensor:
    if labels is None:
        return logits.sum() * 0
    return F.binary_cross_entropy_with_logits(logits, labels.float().view_as(logits))
