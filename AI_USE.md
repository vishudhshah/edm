# AI Use

This file documents how AI tools were used in the DSC 180A Q1 checkpoint (code, experiments, report and README).
The AI used was [Hermes Agent](https://github.com/NousResearch/hermes-agent) as the harness, with the models GLM 5.3 Flash and Claude Opus 5.5.

## Summary

AI was used as an assistant for understanding, debugging, job monitoring and writing polish.
All experimental design, result interpretation and final decisions were mine, and every result in the report was checked by me.

## AI-assisted tasks

### 1. Understanding the EDM paper and codebase

I used AI to help me read Karras et al. (2022) and NVIDIA's EDM code: the preconditioning, the noise schedule, the Heun sampler, and how the training and FID scripts fit together.

### 2. Adapting the code and debugging

I used AI as a coding assistant while adapting the EDM codebase and the course labs to this checkpoint, for example for classifier-free guidance sampling, the subset training runs and the FID evaluation.
It also helped me debug errors in the environment and in the code.
I reviewed every suggestion before using it.

### 3. Running GPU jobs on DSMLP

I ran and tested most of the code myself.
GPUs on DSMLP were often not available for hours, so for the GPU-only jobs I set up an AI agent to watch my place in the queue overnight.
Once a GPU was assigned, it launched the jobs I had prepared and reported the results back to me.
I reviewed those results and logs myself the next morning.

### 4. Report language and structure

I wrote a first draft of every report section and then used AI to refine the language and structure of those drafts.

### 5. README

I used AI to help keep `README.md` up to date as the experiments changed, for example the table that maps each figure to the command that makes it.

### 6. Checking result interpretations

I wrote my own rough interpretations of each result and figure first.
I then used AI to cross-check whether my reading was broadly correct, and I made the final call on every claim in the report.

## Representative prompts

- "Explain how EDM's preconditioning keeps the network's inputs and outputs at unit variance, and where this happens in the code."
- "Walk me through how the Heun sampler in `generate.py` steps through the noise levels."
- "How do I add classifier-free guidance to this sampler using a conditional and an unconditional denoiser call?"
- "`train.py` fails with this error on PyTorch 2.x. RCA and smallest fix?"
- "Watch my DSMLP GPU queue position overnight; when a GPU is assigned, run these prepared jobs and report the results and any errors."
- "Here is my draft of this section. Tighten the language and structure without changing the claims."
- "Here is my interpretation of this figure. Is it broadly correct, and is anything overstated?"
- "Update the README figure table to match the current report figures."

## Human contributions (no AI)

- The direction for the Q2 capstone, decided by our team in a meeting with our mentor.
- The experimental choices: which settings to sweep, which subsets to train on, and which results to report.
- Running and testing most of the code locally and on DSMLP.
- The first draft of every report section, and the first interpretation of every result.
- Most of the LaTeX writing and formatting, done locally and not on Overleaf.

## Testing and validation

- I ran the notebooks and scripts myself and checked their outputs (images, metrics files and logs).
- For the jobs the agent launched overnight, I checked the logs, metrics files and sample images myself before using any result.
- I compared the reported numbers in the tables and text against the saved output files.
- I sanity-checked results against known behavior: near-chance label accuracy without guidance, FID falling during training, and memorization in the small-subset networks shown by nearest-neighbor comparisons.
- I checked the bibliography entries against arXiv and the published proceedings.
