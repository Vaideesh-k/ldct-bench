"""Load the shared settings plus one model's overrides."""

from pathlib import Path

import yaml

# Repo root: this file is ldct/config.py, so go up two levels.
ROOT = Path(__file__).resolve().parent.parent

LOCKED_TEST_PATIENT = "L506"


def _deep_merge(base, override):
    """Merge override into base. Nested dicts are merged key by key;
    any other value in override replaces the one in base."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(model_name, extra=None):
    """Load configs/default.yaml, then apply configs/models/<model_name>.yaml
    on top if it exists, then the optional `extra` yaml file (used by tests to
    point everything at temporary folders)."""
    cfg = _read_yaml(ROOT / "configs" / "default.yaml")

    model_file = ROOT / "configs" / "models" / f"{model_name}.yaml"
    if model_file.exists():
        cfg = _deep_merge(cfg, _read_yaml(model_file))
    if extra:
        cfg = _deep_merge(cfg, _read_yaml(extra))

    cfg["model_name"] = model_name
    check_locked_patient(cfg)
    return cfg


def check_locked_patient(cfg):
    """Guard the locked test patient, even if a model file overrides the splits."""
    for split in ("train_patients", "val_patients"):
        if LOCKED_TEST_PATIENT in cfg["data"].get(split, []):
            raise ValueError(
                f"{LOCKED_TEST_PATIENT} is the locked test patient and must not "
                f"appear in data.{split}"
            )


def resolve(path):
    """Turn a config path (e.g. "data/mayo") into an absolute path under ROOT.
    Paths that are already absolute are returned unchanged."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
