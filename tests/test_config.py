from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "train_example.yaml"


def _load_yaml():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_example_yaml_has_required_top_level_keys():
    cfg = _load_yaml()
    for key in ("experiment", "dataset", "model", "splitting", "training"):
        assert key in cfg, f"example.yaml missing top-level key: {key}"


def test_dataset_config_accepts_yaml():
    from eris2.data import DatasetConfig
    cfg = _load_yaml()
    ds = DatasetConfig(**cfg["dataset"])
    assert ds.window_len == 2 * ds.window_half + 1


def test_model_config_accepts_yaml_and_matches_dataset_dims():
    from eris2.data import DatasetConfig
    from eris2.model import ModelConfig
    cfg = _load_yaml()
    ds = DatasetConfig(**cfg["dataset"])
    mc = ModelConfig(**cfg["model"])
    assert mc.window_len == ds.window_len, (mc.window_len, ds.window_len)
    assert mc.mutation_position == ds.window_half, (mc.mutation_position, ds.window_half)
    assert mc.ca_matrix_size == ds.ca_neigh, (mc.ca_matrix_size, ds.ca_neigh)
    assert mc.atom_matrix_size == ds.atom_neigh, (mc.atom_matrix_size, ds.atom_neigh)


def test_splitting_defaults_to_cluster_aware():
    cfg = _load_yaml()
    s = cfg["splitting"]
    assert s.get("cluster_column"), (
        "example.yaml must ship with a cluster_column set. Random splits leak "
        "homology."
    )
    assert s.get("allow_random_split", False) is False, (
        "example.yaml must not enable allow_random_split by default."
    )
