"""Model registry: build a model from its name.

Each model lives in its own file, ldct/models/<name>.py, with a build() function
(see template.py for the full contract). The pipeline never imports models
directly; it calls build_model("<name>", **model_args).
"""
import importlib
import os

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model(name, **model_args):
    """Import ldct/models/<name>.py and return its build(**model_args)."""
    module_name = f"ldct.models.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name != module_name:
            raise   # the model file exists but needs a package that is not installed
        raise SystemExit(
            f"No model called '{name}': ldct/models/{name}.py does not exist.\n"
            f"To add it: cp ldct/models/template.py ldct/models/{name}.py, fill it in, "
            f"and (optionally) add configs/models/{name}.yaml.\n"
            f"Available models: {', '.join(available_models()) or 'none'}"
        )
    if not hasattr(module, "build"):
        raise SystemExit(f"ldct/models/{name}.py has no build() function (see template.py)")
    return module.build(**model_args)


def available_models():
    """Names of all model files in this folder (template.py is only an example)."""
    return sorted(f[:-3] for f in os.listdir(MODELS_DIR)
                  if f.endswith(".py") and not f.startswith("_") and f != "template.py")
