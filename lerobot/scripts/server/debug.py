import hashlib
import logging
import torch


def summarize_state_dict(state_dict):
    summary = {}
    for key, tensor in state_dict.items():
        tensor_cpu = tensor.detach().cpu()
        mean = tensor_cpu.mean().item()
        std = tensor_cpu.std().item()
        shape = tuple(tensor_cpu.shape)

        # Можно использовать простой хеш для идентификации тензора
        tensor_bytes = tensor_cpu.numpy().tobytes()
        checksum = hashlib.md5(tensor_bytes).hexdigest()[
            :8
        ]  # короткий хеш для удобства чтения

        summary[key] = {
            "shape": shape,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "checksum": checksum,
        }
    return summary


def print_transitions_summary(transitions):
    for transition in transitions:
        print_transition_summary(transition)


def print_transition_summary(transition):
    summary = summarize_transition(transition)
    logging.info(f"[DEBUG] Transition summary: {summary}")


def summarize_transition(transition):
    summary = {}
    for key, value in transition.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": tuple(value.shape),
                "mean": value.mean().item(),
                "std": value.std().item(),
                "checksum": hashlib.md5(value.numpy().tobytes()).hexdigest()[:8],
            }
        else:
            summary[key] = value

    summary["next_state"] = summarize_state_dict(transition["next_state"])
    summary["action"] = {
        "shape": tuple(transition["action"].shape),
        "mean": transition["action"].mean().item(),
        "std": transition["action"].std().item(),
        "checksum": hashlib.md5(transition["action"].numpy().tobytes()).hexdigest()[:8],
    }
    summary["reward"] = transition["reward"]
    summary["done"] = transition["done"]

    return summary


def print_state_summary(summary):
    for key, val in summary.items():
        logging.info(
            f"{key}: shape={val['shape']}, mean={val['mean']}, std={val['std']}, checksum={val['checksum']}"
        )
