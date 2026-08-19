import torch


def translation_goal_error(predicted, target, valid_mask):
    error = (predicted - target).norm(dim=-1)
    valid = valid_mask.squeeze(-1).bool()
    return error[valid].mean() if valid.any() else error.sum() * 0


def direction_cosine_error(predicted, target, valid_mask):
    error = 1 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    valid = valid_mask.squeeze(-1).bool()
    return error[valid].mean() if valid.any() else error.sum() * 0


def gripper_transition_accuracy(predicted, target):
    return (predicted.sign() == target.sign()).float().mean()


def removal_sensitivity(original_actions, removed_actions):
    return (removed_actions - original_actions).flatten(1).norm(dim=-1).mean()
