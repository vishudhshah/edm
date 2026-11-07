# Checkpoint notes: DSC 180A, D32 (EDM + CFG)

My notes for the checkpoint (due Friday, November 6, 2026): what I've done, decisions I made, and my answers to the notebook questions.

## My checklist

- [x] First 24h: repo public, README (setup, downloads, command per figure/table), notes file, CIFAR lab running on GPU (parts 1 to 3 done on DSMLP)
- [x] Generation and sampling: lab parts 1 to 5, with sample grids, trajectories, a CFG sweep over guidance strengths 0, 1, 1.5, 2, 3 and 4, and accuracy, FID and FID_class at 1,000 images per strength
- [x] Small-subset training (part 6): I train on 100 and 1,000 CIFAR-10 images with 10% label dropout, save and measure snapshots, check nearest training neighbors, draw CFG grids
- [x] MNIST CFG network: I train the lab's U-Net from scratch on the 60k digits, replacing the label with 10 ("no label") on a random 10%, then sample at guidance strengths 0, 1, 2 and 4 and measure accuracy and FID on the classifier's features
- [x] (Encouraged) full EDM `train.py` run on a DSMLP GPU for as many hours as I can: loss curve and FID across snapshots
- [x] Report: title, abstract, intro (problem, related work, my data), preliminary results, link to this repo

## Setup decisions

- I check off "First 24h" above: repo, README, notes, and parts 1 to 3 all done on DSMLP.
- I run in the DSMLP default environment (no conda env; Colab as fallback); the notebook downloads about 720 MB (pretrained checkpoints, CIFAR-10, FID refs).
- I apply the one-line `train.py` fix for PyTorch 2.2 and newer, per the notebook's instructions.

## Things I need to remember

1. My FIDs at 1,000 samples will be much higher than the published 50k-sample numbers (EDM: 1.97 unconditional, 1.79 conditional). I only compare runs at equal sample counts.
2. Models trained on 100 or 1,000 images memorize: their samples become copies of training images. So part 6 tests my training pipeline, and the full run is where real training shows.
3. I log seeds and sampling settings for every figure so each one is reproducible from the README table.

## My answers to part 3

From `report/figure/trajectory_w1.png` and `trajectory_w4.png` (cat, seed 2, 3 images, 8 of the 18 noise levels, from 80 down to 0.002):

- When does the class first show in the denoiser's estimate? At noise levels 80 and 28 the estimate is a blur close to the average CIFAR-10 image. A rough silhouette appears around 3.3, and the cat is clearly recognizable at 1.1. With guidance 4 the estimates commit to strong colors earlier (a warm reddish-brown cast at high noise), but only 1 of 3 images is clearly more cat-like at 3.3, so there is no consistent shift.
- From which noise level on does the image barely change? From about 0.3; at 0.06 and below the noisy image and the estimate are almost identical.
- How do the guidance-4 estimates differ at low noise? Higher contrast and saturation, sharper but less natural features (clipped whites, a blocky dark edge in one image). The compositions also differ from guidance 1 despite the same starting noise, because the trajectories split at high noise.

## Key numbers (all at 1,000 samples, 18-step Heun)

Pretrained CIFAR-10 CFG sweep (full sweep in `outputs/cfg_sweep_n1000.json`):

| Guidance | Accuracy | FID | FID_class |
|---|---|---|---|
| 0 | 11.5% | 31.2 | n/a |
| 1 | 95.5% | 31.3 | 112.1 |
| 1.5 | 98.7% | 31.4 | 107.6 |
| 2 | 98.8% | 33.1 | 108.9 |
| 3 | 99.5% | 36.0 | 112.4 |
| 4 | 98.9% | 39.0 | 117.5 |

Per-class FID is lowest at guidance 1.5.

MNIST from-scratch CFG (20k steps, 10% label dropout):

| Guidance | Accuracy | FID |
|---|---|---|
| 0 | 9.8% | 164.9 |
| 1 | 98.3% | 39.4 |
| 2 | 100% | 67.2 |
| 4 | 100% | 122.1 |

- Memorization check (median distance to the nearest training image, lower means memorized): the 100-image subset's samples sit at 1.2 versus 23.7 for real test images (memorized); the 1,000-image subset's at 14.5 versus 21.3 (much less).
- Full train.py run (unconditional, `--cond=0`; 1 GPU, 4 h cap): 1,004 kimg, with snapshot FID 414.7 at kimg 0, 106.2 at 502 and 76.9 at 1,004. Curves: `training_run/`.

## Work log

- 2026-10-29: I forked NVlabs/edm, added the CIFAR-10 EDM lab and MNIST guidance lab, wrote the checkpoint README section (setup, downloads, figure/table map), and started this notes file.
- 2026-10-29: I ran the notebook parts 1 to 3 on DSMLP, added a .gitignore, committed the executed notebook, and synced everything across DSMLP and my Mac. TA check-in is tomorrow.
- 2026-11-04/05 (overnight): I got a DSMLP GPU (GTX 1080 Ti, course queue) and ran everything end-to-end: the full CIFAR lab notebook (parts 1 to 6, \~2h18m), the MNIST from-scratch CFG run (\~27 min), and the full `train.py` run (\~3h, stopped after 3 snapshots, since at least 3 are required). All three jobs ran one after another on one pod.
- 2026-11-05: I copied all artifacts (JSON results, npz sample dumps, subset snapshots, training curves, executed notebook) to the Mac and verified checksums. Snapshot .pkl/.pt checkpoints were not copied (too large). I filled in the report (`report/report.tex`: abstract, methods, results tables and figures, discussion, conclusion) and compiled it to `report/report.pdf` (9 pages). The README table now points at the real output files.
- 2026-11-06: I polished the report: one sentence per line, `\(...\)` math, `[H]` floats, booktabs tables in `report/table/`, new figures exported from the notebook (class grid, trajectories at guidance 1 and 4, CFG sweep plots, nearest-neighbor grids), part 3 answers, a related-work section, and a data section. I fixed facts (the full run was unconditional; 1,004 kimg is 0.5% of the 200M-image schedule) and removed the template's placeholder figures and tables. The report is now 15 pages.
