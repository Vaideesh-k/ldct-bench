"""Check how train.py picks best.pth (lowest training loss, from best_from_epoch on).

Uses made-up loss values only: no data, no model, runs in a second.
"""
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from train import choose_best

failed = []


def check(label, ok):
    print(f"{label}: {bool(ok)}")
    if not ok:
        failed.append(label)


def run(losses, best_from=1, start=None):
    """Feed one loss per epoch (epoch 1, 2, ...) and return the final (best_loss, best_epoch)
    plus the epochs at which best.pth would have been saved. `start` = (best_loss,
    best_epoch, first_epoch) pretends we resumed an earlier run."""
    best_loss, best_epoch, first = start or (math.inf, 0, 1)
    saved = []
    for epoch, loss in enumerate(losses[first - 1:], start=first):
        best_loss, best_epoch, improved = choose_best(epoch, loss, best_loss, best_epoch, best_from)
        if improved:
            saved.append(epoch)
    return best_loss, best_epoch, saved


# like CT-Mamba: falls for 10 epochs, jumps up at epoch 11 (2 extra loss terms), falls
# again but never back below the epoch-10 value
ctm = [1.0 - 0.05 * e for e in range(10)] + [0.70, 0.68, 0.66, 0.64, 0.62]
check("loss values as intended (epoch 11+ always above epoch 10)", min(ctm[10:]) > ctm[9])

_, best, _ = run(ctm, best_from=1)
check("old rule (best_from 1): stuck at epoch 10", best == 10)

loss, best, saved = run(ctm, best_from=11)
check("best_from 11: best is the lowest of epochs 11+ (epoch 15)", best == 15 and loss == ctm[14])
check("best_from 11: best.pth still saved during epochs 1-10", saved[:10] == list(range(1, 11)))
check("best_from 11: saved at epoch 11 even though its loss is higher", 11 in saved)

_, best, _ = run(ctm[:5], best_from=11)
check("run shorter than best_from (e.g. smoke): best = lowest so far", best == 5)

# resuming a run that was started with the old rule: its best is epoch 10
_, best, _ = run(ctm, best_from=11, start=(ctm[9], 10, 14))
check("resume at epoch 14 with an old best at epoch 10: best moves to 11+", best == 15)
_, best, _ = run(ctm, best_from=11, start=(0.30, 12, 14))
check("resume with a best already at epoch 12: kept unless beaten", best == 12)

# the normal case: the loss keeps falling, best_from changes nothing
falling = [1.0 / e for e in range(1, 21)]
check("falling loss: same result with best_from 1 and 11",
      run(falling, 1)[1] == run(falling, 11)[1] == 20)

sys.exit(1 if failed else 0)
