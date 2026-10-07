"""Quick checks for ldct/models: the registry, the template and RED-CNN."""
import os
import sys

import torch

# let this file import from the ldct package (it lives one folder up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import load_config
from ldct.models import available_models, build_model

failed = []


def check(label, ok):
    """Print one result and remember failures (the script exits with code 1 at the end)."""
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# RED-CNN, built the way train.py builds it (model_args from its config)
cfg = load_config("redcnn")
red = build_model("redcnn", **cfg["model_args"])
print(f"RED-CNN parameters: {n_params(red):,}")
check("RED-CNN has 1.85 M parameters", round(n_params(red) / 1e6, 2) == 1.85)
check("redcnn config: lr 1e-4, lr_min 1e-5", cfg["train"]["lr"] == 1e-4 and cfg["train"]["lr_min"] == 1e-5)
with torch.no_grad():
    for shape in [(4, 1, 64, 64), (1, 1, 100, 77)]:
        x = torch.rand(shape)
        y = red(x)
        check(f"RED-CNN keeps shape {shape}", y.shape == x.shape)

# template: a working example
tiny = build_model("template")
check("template builds and runs", tiny(torch.rand(2, 1, 64, 64)).shape == (2, 1, 64, 64))
check("template not listed as a real model", "template" not in available_models())
print("available models:", available_models())

# missing model -> clear message pointing at template.py
try:
    build_model("does_not_exist")
    check("missing model gives a message", False)
except SystemExit as e:
    check("missing model message mentions template.py", "template.py" in str(e))

sys.exit(1 if failed else 0)
