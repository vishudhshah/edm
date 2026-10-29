# Checkpoint notes — DSC 180A · D32 (EDM + CFG)

My notes for the checkpoint (due Friday, November 6, 2026): what I've done, decisions I made, and my answers to the notebook questions.

## My checklist

- [ ] First 24h: I make the repo public, write the README (setup, downloads, command per figure/table), keep this notes file, and get the CIFAR lab running on a GPU (parts 1–3)
- [ ] Generation & sampling: lab parts 1–5 — sample grids, trajectories, CFG sweep w ∈ {0, 1, 1.5, 2, 3, 4}, accuracy/FID/FID_class at 1,000 images per w
- [ ] Small-subset training (part 6): I train on 100 and 1,000 CIFAR-10 images with 10% label dropout, save and measure snapshots, check nearest training neighbors, draw CFG grids
- [ ] MNIST CFG network: I train the lab's U-Net from scratch on the 60k digits, replacing the label with 10 ("no label") on a random 10%, then sample at w ∈ {0, 1, 2, 4} and measure accuracy + FID on the classifier's features
- [ ] (Encouraged) full EDM `train.py` run on a DSMLP GPU for as many hours as I can: loss curve + FID across snapshots
- [ ] Report: title, abstract, intro (problem, related work, my data), preliminary results, link to this repo

## Setup decisions

- I use the repo's `environment.yml` (`conda env create -f environment.yml -n edm`).
- I run on a DSMLP GPU (Colab as fallback); the notebook downloads ~720 MB (pretrained checkpoints, CIFAR-10, FID refs).
- I apply the one-line `train.py` fix for PyTorch ≥ 2.2, per the notebook's instructions.

## Things I need to remember

1. My FIDs at 1,000 samples will be much higher than the published 50k-sample numbers (EDM: 1.97 uncond / 1.79 cond). I only compare runs at equal sample counts.
2. Models trained on 100 or 1,000 images memorize: their samples become copies of training images. So part 6 tests my training pipeline, and the full run is where real training shows.
3. I log seeds and sampling settings for every figure so each one is reproducible from the README table.

## My answers to part 3

- (To fill in after I run the trajectory section.)

## Work log

- 2026-10-29: I forked NVlabs/edm, added the CIFAR-10 EDM lab and MNIST guidance lab, wrote the checkpoint README section (setup, downloads, figure/table map), and started this notes file.
