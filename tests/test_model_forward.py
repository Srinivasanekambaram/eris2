import torch


def _make_dummy_batch(batch_size: int, cfg) -> dict:
    B = batch_size
    W = cfg.window_len
    return {
        "wt_embedding": torch.randn(B, W, cfg.esm_dim),
        "mut_embedding": torch.randn(B, W, cfg.esm_dim),
        "ca_distance_matrix": torch.rand(B, cfg.ca_matrix_size, cfg.ca_matrix_size),
        "atom_distance_matrix": torch.rand(B, cfg.atom_matrix_size, cfg.atom_matrix_size),
        "rsa_values": torch.rand(B, W),
        "backbone_angles": torch.rand(B, W, 2),
        "hbond_features": torch.rand(B, W, 3),
        "ss_features": torch.zeros(B, W, 8),
        "charge_features": torch.rand(B, W, 1),
        "hydrophobicity_features": torch.rand(B, W, 1),
        "atom_features": torch.rand(B, W, 4),
        "window_mask": torch.ones(B, W),
    }


def test_forward_default_config():
    from eris2.model import DDGPredictor, ModelConfig
    cfg = ModelConfig()
    model = DDGPredictor(cfg).eval()
    batch = _make_dummy_batch(batch_size=2, cfg=cfg)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2,), f"expected [B] output, got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite values in model output"


def test_forward_batch_size_1():
    from eris2.model import DDGPredictor, ModelConfig
    cfg = ModelConfig()
    model = DDGPredictor(cfg).eval()
    batch = _make_dummy_batch(batch_size=1, cfg=cfg)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (1,)


def test_state_dict_roundtrip():
    from eris2.model import DDGPredictor, ModelConfig
    cfg = ModelConfig()
    m1 = DDGPredictor(cfg)
    m2 = DDGPredictor(cfg)
    batch = _make_dummy_batch(batch_size=2, cfg=cfg)
    m1.eval(); m2.eval()
    with torch.no_grad():
        y1_a = m1(batch)
        y2_a = m2(batch)
    assert not torch.allclose(y1_a, y2_a), "randomly initialised models tied"
    m2.load_state_dict(m1.state_dict())
    with torch.no_grad():
        y1_b = m1(batch)
        y2_b = m2(batch)
    assert torch.allclose(y1_b, y2_b, atol=1e-6), "state_dict roundtrip mismatch"
