# Reproducing EDM on CIFAR-10: Conditioning, Classifier-Free Guidance, and FID

**Author:** Vishudh Shah, DSC 180A Section D32, UC San Diego, Fall 2026<br>
**Mentor:** Prof. Yusu Wang

## Overview

This repository holds my Quarter 1 checkpoint for DSC 180A.
It is a fork of NVIDIA's [EDM](https://github.com/NVlabs/edm) codebase, with the course labs and my experiments added on top.

What I did:
- Sampled CIFAR-10 images from NVIDIA's pretrained unconditional and class-conditional EDM models.
- Added classifier-free guidance (CFG) and swept the guidance weight over 0, 1, 1.5, 2, 3 and 4, measuring label accuracy, FID and per-class FID.
- Looked inside the sampler: the noisy image and the denoiser's estimate at each noise level.
- Trained small CFG networks on 100 and 1,000 CIFAR-10 images and checked them for memorization.
- Trained a CFG network on MNIST from scratch.
- Ran EDM's own `train.py` on CIFAR-10 for about 3 hours on one GPU.

The report is at [`report/report.pdf`](report/report.pdf).
My work log and answers to the notebook questions are in [`NOTES.md`](NOTES.md).

## Repository structure

| Path | Contents |
| :--- | :--- |
| `dsc180-d32-cifar10-edm-lab/` | Main notebook `cifar10_edm_lab.ipynb` (parts 1 to 6) and its saved results in `outputs/` |
| `dsc180-d32-mnist-guidance-lab/` | MNIST guidance lab, my from-scratch script `mnist_cfg_from_scratch.py`, and its results in `mnist_cfg_outputs/` |
| `training_run/` | Log, loss curve and snapshot FIDs from the full `train.py` run |
| `report/` | LaTeX source `report.tex`, figures, tables, bibliography and the compiled `report.pdf` |
| `NOTES.md` | Checklist, decisions, part 3 answers, key numbers and work log |
| `AI_USE.md` | How AI tools were used |
| `train.py`, `generate.py`, `fid.py`, `dataset_tool.py`, `training/`, ... | NVIDIA's original EDM code |

## Setup

I ran everything on UCSD's DSMLP in the default environment, whose Python and Jupyter already have what the notebooks need.
Any recent Python with PyTorch, NumPy, SciPy and Matplotlib should also work.

| Job | Hardware | Time (GTX 1080 Ti) |
| :--- | :--- | :--- |
| CIFAR-10 lab, parts 1 to 6 | GPU (also runs on Apple GPUs, since sampling is in float32) | about 2.3 hours |
| MNIST from-scratch script | GPU recommended | about 30 minutes |
| Full `train.py` run | GPU | stopped after about 3 hours |

`train.py` needs a one-line fix for PyTorch 2.2 and newer.
The notebook explains it and applies it to its own copy of EDM.

## Data

The notebooks download what they need on first run (about 720 MB in total).

| What | Source | Used by |
| :--- | :--- | :--- |
| Pretrained EDM CIFAR-10 models (unconditional and conditional, VP) | `https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/` | CIFAR-10 lab |
| CIFAR-10 (50,000 train, 10,000 test images) | Hugging Face copy `uoft-cs/cifar10` (an existing `cifar-10-python.tar.gz` in `downloads/` is used instead) | CIFAR-10 lab, `train.py` |
| CIFAR-10 FID reference statistics | `https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz` | CIFAR-10 lab, `train.py` |
| MNIST (60,000 train, 10,000 test digits) | `torchvision.datasets.MNIST` | MNIST script |
| Frozen MNIST digit classifier and lab weights | included in `dsc180-d32-mnist-guidance-lab/` | MNIST script |

For the full training run, the CIFAR-10 lab's "actual training run" cell writes CIFAR-10 in EDM's ZIP format (the same format `dataset_tool.py` makes).
Save it as `datasets/cifar10-32x32.zip`, the path the training command below uses.

## Reproducing figures and tables

To run the CIFAR-10 lab, open `dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb` and run all cells.

| Report item | How to produce it | Output |
| :--- | :--- | :--- |
| Sample grids from the pretrained models | CIFAR-10 lab, parts 1 and 2 | `report/figure/cond_grid_classes.png` |
| CFG sample grids at guidance 0 and 4 | CIFAR-10 lab, part 4 | `outputs/cfg_w*_n1000.npz`, `report/figure/cifar_grid_w0.png`, `cifar_grid_w4.png` |
| Sampling trajectories (noisy image and denoiser estimate) | CIFAR-10 lab, parts 3 and 4 | `report/figure/trajectory_w1.png`, `trajectory_w4.png`, `outputs/trajectory_cfg_w4.gif` |
| CFG sweep table and plots (accuracy, FID, per-class FID) | CIFAR-10 lab, part 5 | `outputs/cfg_sweep_n1000.json`, `report/table/cfg_sweep.tex`, `report/figure/cfg_sweep_plots.png` |
| Small-subset training: loss and FID across snapshots | CIFAR-10 lab, part 6 | `report/figure/subset_loss_fid.png`, `report/table/subset_fid.tex` |
| Nearest training images (memorization check) | CIFAR-10 lab, part 6 | `report/figure/nearest_subset100.png`, `nearest_subset1000.png` |
| CFG grids from the small-subset networks | CIFAR-10 lab, part 6 | `report/figure/subset100_cfg.png`, `subset1000_cfg.png` |
| MNIST CFG: accuracy, FID and sample grids | `cd dsc180-d32-mnist-guidance-lab && python3 mnist_cfg_from_scratch.py` | `mnist_cfg_outputs/mnist_cfg_results.json`, `report/table/mnist_cfg.tex`, `report/figure/mnist_grid_w0.png`, `mnist_grid_w1.png` |
| Full training run: loss curve and FID across snapshots | `torchrun --standalone --nproc_per_node=1 train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=0 --arch=ddpmpp` | `training_run/loss_curve.png`, `fid_vs_kimg.png`, `snapshot_fid.json` |

To rebuild the report, run `latexmk -lualatex report.tex` inside `report/`.

## A note on FID

Every FID in this project uses 1,000 generated images per setting, so the values are much higher than the paper's 50,000-image results (1.97 unconditional, 1.79 conditional).
I only compare runs at equal sample counts.
The MNIST FID uses the lab classifier's features against the 10,000 MNIST test images, so it is only comparable across guidance weights within the same run.

## AI use

I used [Hermes Agent](https://github.com/NousResearch/hermes-agent) with GLM 5.3 Flash and Claude Opus 5.5 to help me understand the EDM paper and code, adapt and debug the code, monitor and launch GPU jobs on DSMLP overnight, polish the report's language from my own drafts, keep this README current, and cross-check my interpretations of results.
I made the experimental design, the first drafts, the result interpretations, most of the LaTeX and all final decisions myself, and our team chose the capstone direction with our mentor.
See [`AI_USE.md`](AI_USE.md) for the full summary, representative prompts and how results were validated.

## Credits and license

The EDM code and pretrained models are by Tero Karras, Miika Aittala, Timo Aila and Samuli Laine at NVIDIA ([paper](https://arxiv.org/abs/2206.00364)), released under the [CC BY-NC-SA 4.0 license](http://creativecommons.org/licenses/by-nc-sa/4.0/) (see [`LICENSE.txt`](LICENSE.txt)).
The CIFAR-10 and MNIST labs are DSC 180A course material.
