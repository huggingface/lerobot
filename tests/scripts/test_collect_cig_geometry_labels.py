import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "examples/cig_vla/collect_cig_geometry_labels.py"
spec = importlib.util.spec_from_file_location("collect_cig_geometry_labels", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_entrypoint_is_schema_inspection_only():
    assert module.main is not None
    assert not hasattr(module, "cig_dataset_features")
    assert not hasattr(module, "dry_run")
    assert not hasattr(module, "generate_geometry_labels")
