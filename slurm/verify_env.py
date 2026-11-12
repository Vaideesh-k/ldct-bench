import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| CUDA build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
from mamba_ssm import Mamba
import mamba_ssm, causal_conv1d
print("mamba_ssm", mamba_ssm.__version__, "| causal_conv1d", causal_conv1d.__version__)
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 64, 16, device="cuda")
y = m(x); y.sum().backward()
print("Mamba forward+backward on GPU:", tuple(y.shape), "finite:", bool(torch.isfinite(y).all()))
import timm, einops, pywt, ptwt, transformers, natsort, scipy, skimage, pydicom, yaml
print("other imports OK")
