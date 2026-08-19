from lerobot.benchmarks.cig_bench.libero_backend import LiberoSafetyOnlineBackend


def test_online_evaluator_does_not_expose_training_geometry_labels():
    evaluator = LiberoSafetyOnlineBackend()
    assert not hasattr(evaluator, "object_registry")
    assert not hasattr(evaluator, "training_geometry_labels")
