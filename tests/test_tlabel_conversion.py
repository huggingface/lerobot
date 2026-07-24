"""Tests for TLabel to LeRobot conversion."""
import json
from pathlib import Path

import pytest

# Skip all tests if tlabel or lerobot not installed
tlabel = pytest.importorskip("tlabel")
lerobot = pytest.importorskip("lerobot")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "tlabel"))

from convert_tlabel_to_lerobot import (
    DEFAULT_TACTILE_FEATURES,
    SENSOR_CONFIGS,
    build_features,
    extract_tactile_features,
    _load_manual,
    _load_csv_episodes,
)


class TestFeatureBuilding:
    """Test feature dict construction."""

    def test_default_features(self):
        """Default features include all 8 tactile groups + base features."""
        features = build_features("gelsight")
        assert "timestamp" in features
        assert "episode_index" in features
        assert "frame_index" in features
        assert "index" in features
        assert "task_index" in features
        assert "observation.tactile.contact" in features
        assert "observation.tactile.force" in features
        assert "observation.tactile.deformation" in features
        assert "observation.tactile.slip" in features
        assert "observation.tactile.texture" in features
        assert "observation.tactile.contact_geometry" in features
        assert "observation.tactile.field" in features
        assert "observation.tactile.dynamics" in features

    def test_gelsight_has_image(self):
        """GelSight sensor includes tactile image feature."""
        features = build_features("gelsight", has_image=True)
        assert "observation.images.tactile" in features
        assert features["observation.images.tactile"]["dtype"] == "video"

    def test_paxini_no_image(self):
        """PaXini sensor does not include image feature."""
        features = build_features("paxini", has_image=False)
        assert "observation.images.tactile" not in features

    def test_custom_config(self, tmp_path):
        """Custom YAML config overrides defaults."""
        config = tmp_path / "custom.yaml"
        config.write_text(
            "observation.tactile.contact:\n"
            "  dtype: float32\n"
            "  shape: [1]\n"
            "  names: [contact]\n"
        )
        features = build_features("gelsight", config_path=str(config))
        assert "observation.tactile.contact" in features


class TestFeatureExtraction:
    """Test TLabel frame to LeRobot feature extraction."""

    def test_basic_extraction(self):
        """Extract features from a minimal frame."""
        frame = {
            "contact": 1.0,
            "force_magnitude": 2.5,
            "force_direction": 0.3,
            "force_peak": 3.1,
            "deformation_magnitude": 0.1,
            "temporal_deformation_rate": 0.05,
            "slip_entropy": 0.8,
            "slip_event": 1.0,
            "texture_energy": 0.2,
            "contact_area": 100.0,
            "centroid_x": 50.0,
            "centroid_y": 75.0,
            "normal_mag": 1.5,
            "normal_var": 0.1,
            "shear_mag": 0.3,
            "shear_dir": 1.2,
            "delta_normal": 0.2,
            "delta_shear": 0.1,
            "friction_cone_ratio": 0.7,
        }
        result = extract_tactile_features(frame, "gelsight")

        assert result["observation.tactile.contact"] == [1.0]
        assert result["observation.tactile.force"] == [2.5, 0.3, 3.1]
        assert result["observation.tactile.deformation"] == [0.1, 0.05]
        assert result["observation.tactile.slip"] == [0.8, 1.0]
        assert result["observation.tactile.texture"] == [0.2]
        assert result["observation.tactile.contact_geometry"] == [100.0, 50.0, 75.0]
        assert result["observation.tactile.field"] == [1.5, 0.1, 0.3, 1.2]
        assert result["observation.tactile.dynamics"] == [0.2, 0.1, 0.7]

    def test_missing_values_default_to_zero(self):
        """Missing fields default to 0.0."""
        frame = {"contact": 1.0}
        result = extract_tactile_features(frame, "paxini")

        assert result["observation.tactile.contact"] == [1.0]
        assert result["observation.tactile.force"] == [0.0, 0.0, 0.0]
        assert result["observation.tactile.deformation"] == [0.0, 0.0]

    def test_empty_frame(self):
        """Empty frame returns all zeros."""
        result = extract_tactile_features({}, "gelsight")
        for key, value in result.items():
            assert all(v == 0.0 for v in value), f"{key} has non-zero values: {value}"


class TestSensorConfigs:
    """Test sensor-specific configurations."""

    def test_known_sensors(self):
        """All documented sensors are in config."""
        expected = {"gelsight", "digit", "paxini", "daimon", "touchd", "univtac", "vtac"}
        assert set(SENSOR_CONFIGS.keys()) == expected

    def test_visual_sensors_have_images(self):
        """Visual tactile sensors have image capability."""
        assert SENSOR_CONFIGS["gelsight"]["has_image"] is True
        assert SENSOR_CONFIGS["digit"]["has_image"] is True

    def test_force_sensors_no_images(self):
        """Force-based sensors have no image capability."""
        assert SENSOR_CONFIGS["paxini"]["has_image"] is False
        assert SENSOR_CONFIGS["daimon"]["has_image"] is False


class TestDataLoading:
    """Test manual data loading fallback."""

    def test_load_json(self, tmp_path):
        """Load from tlabel_export.json."""
        data = {
            "metadata": {"sensor": "gelsight", "fps": 30},
            "frames": [
                {"episode_index": 0, "contact": 1.0, "force_magnitude": 2.0},
                {"episode_index": 0, "contact": 0.0, "force_magnitude": 0.0},
                {"episode_index": 1, "contact": 1.0, "force_magnitude": 1.5},
            ],
        }
        json_file = tmp_path / "tlabel_export.json"
        json_file.write_text(json.dumps(data))

        result = _load_manual(tmp_path)
        assert len(result["episodes"]) == 2
        assert len(result["episodes"][0]) == 2
        assert len(result["episodes"][1]) == 1

    def test_load_csv(self, tmp_path):
        """Load from CSV files."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "episode_index,contact,force_magnitude\n"
            "0,1.0,2.5\n"
            "0,0.0,0.0\n"
            "1,1.0,1.0\n"
        )

        result = _load_csv_episodes([csv_file])
        assert len(result["episodes"]) == 2
        assert result["episodes"][0][0]["contact"] == 1.0

    def test_no_data_raises(self, tmp_path):
        """Missing data files raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_manual(tmp_path)
