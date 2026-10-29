"""Checkpoint item: an MNIST CFG network trained from scratch.

Trains the MNIST guidance lab's U-Net on all 60,000 training digits with 10% label dropout (the label becomes
10, "no request"), then samples with w in {0, 1, 2, 4} and measures, for each w, the label accuracy from the
lab's frozen digit classifier and FID on that classifier's 128-d features against real images.

Run on a GPU pod:  python3 mnist_cfg_from_scratch.py
Everything lands in dsc180-d32-mnist-guidance-lab/mnist_cfg_outputs/ .
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
OUT = HERE / "mnist_cfg_outputs"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device(os.environ.get("LAB_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))

STEPS = int(os.environ.get("MNIST_STEPS", 20_000))   # training updates
BATCH = 256
LR = 2e-4
LABEL_DROPOUT = 0.1          # fraction of examples whose label becomes 10 ("no request")
N_EVAL = 1000                # samples per w
WS = [0, 1, 2, 4]

# ---------------------------------------------------------------- the lab's helpers (UNet, sampler, evaluator)
helper = HERE / "mnist_guidance_lab.py"
if not helper.exists():      # write the helper from the notebook's %%writefile cell
    import re
    src = "".join(next(c for c in json.load(open(HERE / "guidance_mnist.ipynb"))["cells"]
                       if c["cell_type"] == "code" and "%%writefile mnist_guidance_lab.py" in "".join(c["source"])
                       )["source"])
    src = src.split("\n", 1)[1]          # drop the %%writefile line
    helper.write_text(src)
    print("wrote", helper)

sys.path.insert(0, str(HERE))
import mnist_guidance_lab as mg          # noqa: E402

torch.manual_seed(0)

# ---------------------------------------------------------------- data (torchvision, as the assignment says)
try:
    from torchvision.datasets import MNIST
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "torchvision"])
    from torchvision.datasets import MNIST

train = MNIST(str(HERE / "data"), train=True, download=True)
test = MNIST(str(HERE / "data"), train=False, download=True)
x1_all = (train.data.float() / 255).unsqueeze(1) * 2 - 1     # clean images in [-1, 1], shape (60000, 1, 28, 28)
y_all = train.targets.clone()
y_all[torch.rand(len(y_all)) < LABEL_DROPOUT] = 10           # 10% of the labels become "no request"
x_test = (test.data.float() / 255).unsqueeze(1) * 2 - 1
print(f"{len(x1_all)} training digits, {int((y_all == 10).sum())} of them carry the no-request label 10")

# ---------------------------------------------------------------- train
net = mg.UNet().to(DEVICE).train()          # same architecture as the lab's frozen generator
opt = torch.optim.Adam(net.parameters(), lr=LR)
losses = []
t0 = time.time()
for step in range(1, STEPS + 1):
    idx = torch.randint(len(x1_all), (BATCH,))
    x1, y = x1_all[idx].to(DEVICE), y_all[idx].to(DEVICE)
    t = torch.rand(len(x1), device=DEVICE)                    # t = 0 noise, t = 1 data, on the cosine path
    a, s = mg.alpha_sigma(t)
    xt = a[:, None, None, None] * x1 + s[:, None, None, None] * torch.randn_like(x1)
    loss = F.mse_loss(net(xt, t, y), x1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 500 == 0 or step == 1:
        print(f"step {step}/{STEPS}  loss {np.mean(losses[-500:]):.4f}  "
              f"({(time.time() - t0) / step * (STEPS - step) / 60:.0f} min left)", flush=True)

torch.save({"state": net.state_dict(), "ch": 64}, OUT / "mnist_cfg_net.pt")
print(f"training done in {(time.time() - t0) / 60:.0f} min")

# ---------------------------------------------------------------- measure: accuracy and FID per w
evaluator = mg.Evaluator().to(DEVICE)
ck = torch.load(HERE / "evaluator.pt", map_location="cpu", weights_only=True)
evaluator.load_state_dict(ck["state"])
evaluator.eval().requires_grad_(False)


@torch.no_grad()
def feats(images):
    """128-d evaluator features; the evaluator consumes images in [-1, 1], as in the lab."""
    out = []
    for i in range(0, len(images), 500):
        _, f = evaluator(images[i:i + 500].to(DEVICE))
        out.append(f.double().cpu())
    return torch.cat(out)


@torch.no_grad()
def classify(images):
    """The evaluator's predicted digit for each image."""
    out = []
    for i in range(0, len(images), 500):
        logits, _ = evaluator(images[i:i + 500].to(DEVICE))
        out.append(logits.argmax(1).cpu())
    return torch.cat(out)


def gaussian(f):
    return f.mean(0), torch.cov(f.T)


def frechet(mu1, s1, mu2, s2):
    e, V = torch.linalg.eigh(s1)
    r = V @ torch.diag(e.clamp(min=0).sqrt()) @ V.T
    m = r @ s2 @ r
    root = torch.linalg.eigvalsh((m + m.T) / 2).clamp(min=0).sqrt().sum()
    return float(((mu1 - mu2) ** 2).sum() + s1.trace() + s2.trace() - 2 * root)


MU_REF, SIG_REF = gaussian(feats(x_test))                   # the 10,000 real test images are the reference
results = {}
for w in WS:
    labels = torch.arange(N_EVAL) % 10                      # balanced requests
    images, nfe = mg.sample(net, N_EVAL, labels, str(DEVICE), steps=16, w=float(w), seed=40)
    pred, f = classify(images), feats(images)               # evaluator expects [-1, 1] images
    acc = (pred == labels).float().mean().item()
    results[w] = {"accuracy": acc, "fid": frechet(*gaussian(f), MU_REF, SIG_REF), "nfe": nfe}
    print(f"w = {w}  accuracy {acc:6.1%}  FID {results[w]['fid']:7.2f}", flush=True)
    np.savez(OUT / f"mnist_cfg_w{w}.npz", images=((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).numpy(),
             labels=labels.numpy())

(OUT / "mnist_cfg_results.json").write_text(json.dumps(
    {"steps": STEPS, "batch": BATCH, "lr": LR, "label_dropout": LABEL_DROPOUT, "n_eval": N_EVAL,
     "results": {str(w): r for w, r in results.items()}}, indent=2))
print("saved results to", OUT / "mnist_cfg_results.json")
