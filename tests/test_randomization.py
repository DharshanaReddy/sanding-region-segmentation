"""Unit tests for pure-Python randomization sampling — no Blender required."""

import yaml

from data_gen.randomization import sample_params

CONFIG_PATH = "data_gen/configs/randomization.yaml"


def _load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_sample_params_is_deterministic():
    cfg = _load_cfg()
    a = sample_params(cfg, index=3, base_seed=42)
    b = sample_params(cfg, index=3, base_seed=42)
    assert a.to_json_dict() == b.to_json_dict()


def test_sample_params_differs_across_index():
    cfg = _load_cfg()
    a = sample_params(cfg, index=0, base_seed=42)
    b = sample_params(cfg, index=1, base_seed=42)
    assert a.to_json_dict() != b.to_json_dict()


def test_sample_params_respects_config_bounds():
    cfg = _load_cfg()
    for index in range(25):
        params = sample_params(cfg, index=index, base_seed=1)
        assert cfg["defects"]["count"]["min"] <= len(params.defect_patches) <= cfg["defects"]["count"]["max"]
        assert cfg["panel"]["roughness"]["min"] <= params.roughness <= cfg["panel"]["roughness"]["max"]
        assert (
            cfg["camera"]["distance_m"]["min"]
            <= params.camera_distance_m
            <= cfg["camera"]["distance_m"]["max"]
        )
        for patch in params.defect_patches:
            assert patch.kind in cfg["defects"]["kinds"]
            assert 0.0 <= patch.u <= 1.0
            assert 0.0 <= patch.v <= 1.0
