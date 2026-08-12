

def test_import_model():
    from eris2.model import DDGPredictor, ModelConfig, count_parameters


def test_import_dataset():
    from eris2.data import (
        DatasetConfig, ProteinDataset, calculate_hbonds, calculate_dssp_features,
    )


def test_import_training():
    from eris2.training import (
        collate_skip_none, evaluate_model, train_model,
    )


def test_import_inference():
    from eris2.inference import collate, load_models, predict_batch


def test_import_metrics():
    from eris2.metrics import full_report


def test_hbond_signature_no_angle_cutoff():
    import inspect
    from eris2.data import calculate_hbonds
    sig = inspect.signature(calculate_hbonds)
    assert "angle_cutoff_deg" not in sig.parameters
