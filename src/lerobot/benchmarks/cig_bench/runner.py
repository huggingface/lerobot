import json
from pathlib import Path


def serialize_results(results, path):
    serializable = {
        key: value.detach().cpu().tolist() if hasattr(value, "detach") else value
        for key, value in results.items()
    }
    Path(path).write_text(json.dumps(serializable, indent=2))
