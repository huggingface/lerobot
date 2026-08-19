import torch


def action_response_magnitude(original, intervened):
    return (intervened - original).flatten(1).norm(dim=-1)


def intervention_response_ratio(original, intervened, geometry_offset, eps=1e-8):
    return action_response_magnitude(original, intervened) / (geometry_offset.norm(dim=-1) + eps)


def intervention_direction_alignment(original_translation, intervened_translation, geometry_offset):
    return torch.nn.functional.cosine_similarity(
        intervened_translation - original_translation, geometry_offset, dim=-1
    )


def motion_suppression(original_motion, intervened_motion):
    return original_motion - intervened_motion
