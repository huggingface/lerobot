import hashlib
import logging


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


def print_state_summary(summary):
    for key, val in summary.items():
        logging.info(
            f"{key}: shape={val['shape']}, mean={val['mean']}, std={val['std']}, checksum={val['checksum']}"
        )
