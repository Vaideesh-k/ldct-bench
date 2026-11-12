"""Time CT-Mamba training steps on this GPU (no checkpoints written)."""
import os, sys, time, torch
sys.path.insert(0, os.path.expanduser("~/ldct-bench")); sys.path.insert(0, os.path.expanduser("~/ldct-bench/scripts"))
from train import make_loaders, training_loss, validate
from ldct.config import load_config
from ldct.models import build_model
cfg = load_config("ct_mamba"); dev = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()
model = build_model("ct_mamba").to(dev); model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
tl, vl = make_loaders(cfg, dev)
print("steps per epoch:", len(tl))
it = iter(tl)
def run(epoch, n, warm=5):
    model.epoch = epoch
    for i in range(warm + n):
        if i == warm: torch.cuda.synchronize(); t = time.time()
        x, y = next(it)
        if i == 0: print("batch shape:", tuple(x.shape))
        loss = training_loss(model, x.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); return (time.time() - t) / n
a = run(1, 30); b = run(11, 30)
print(f"sec/step epochs 1-10: {a:.3f} | epochs 11+: {b:.3f}")
t = time.time(); validate(model, vl, cfg, dev); v = time.time() - t
print(f"validation pass: {v:.1f}s ({len(vl)} slices)")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB of {torch.cuda.get_device_properties(0).total_memory/2**30:.1f}")
n = len(tl); e1, e2 = n * a, n * b
total = 10 * e1 + 240 * e2 + 50 * v
print(f"epoch ~{e1/60:.1f} min (1-10), ~{e2/60:.1f} min (11+); 250 epochs ~{total/3600:.1f} h = {total/86400:.1f} days")
