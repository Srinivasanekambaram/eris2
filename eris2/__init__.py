from pathlib import Path

__version__ = "3.0"

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = str(_ROOT / "model" / "eris2_ensemble.pt")
DEFAULT_CONFIG = str(_ROOT / "model" / "model_config.yaml")
