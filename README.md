<!-- # FD-Shifts -->
<p align="center"><img src="./docs/fd_shifts_logo.svg"></p>

---

<p align="center"><img src="./docs/new_overview.png"></p>

> Reliable application of machine learning-based decision systems in the wild is
> one of the major challenges currently investigated by the field. A large portion
> of established approaches aims to detect erroneous predictions by means of
> assigning confidence scores. This confidence may be obtained by either
> quantifying the model's predictive uncertainty, learning explicit scoring
> functions, or assessing whether the input is in line with the training
> distribution. Curiously, while these approaches all state to address the same
> eventual goal of detecting failures of a classifier upon real-life application,
> they currently constitute largely separated research fields with individual
> evaluation protocols, which either exclude a substantial part of relevant
> methods or ignore large parts of relevant failure sources. In this work, we
> systematically reveal current pitfalls caused by these inconsistencies and
> derive requirements for a holistic and realistic evaluation of failure
> detection. To demonstrate the relevance of this unified perspective, we present
> a large-scale empirical study for the first time enabling benchmarking
> confidence scoring functions w.r.t all relevant methods and failure sources. The
> revelation of a simple softmax response baseline as the overall best performing
> method underlines the drastic shortcomings of current evaluation in the
> abundance of publicized research on confidence scoring.

If you use fd-shifts please cite our [paper]()

```
```

## Table Of Contents

<!--toc:start-->

- [FD-Shifts](#fd-shifts)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data Folder Requirements](#data-folder-requirements)
    - [Training](#training)
    - [Inference](#inference)
    - [Analysis](#analysis)
  - [Acknowledgements](#acknowledgements)

<!--toc:end-->

## Installation

It is recommended to install FD-Shifts in its own environment (venv, conda environment, ...).

1. **Install an appropriate version of [PyTorch](https://pytorch.org/).** Check that CUDA is
   available and that the CUDA toolkit version is compatible with your hardware.

2. **Install FD-Shifts.** This will pull in all dependencies including some version of PyTorch, it
   is strongly recommended that you install a compatible version of PyTorch beforehand. This will
   also make the `fd_shifts` cli available to you.
   ```bash
   pip install https://github.com/iml-dkfz/failure-detection-benchmark
   ```

## Usage

To use `fd_shifts` you need to set the following environment variables

```bash
export EXPERIMENT_ROOT_DIR=/absolute/path/to/your/experiments
export DATASET_ROOT_DIR=/absolute/path/to/datasets
```

Alternatively, you may write them to a file and source that before running
`fd_shifts`, e.g.

```bash
mv example.env .env
```

Then edit `.env` to your needs and run

```bash
source .env
```

### Data Folder Requirements

For the predefined experiments we expect the data to be in the following folder
structure relative to the folder you set for `$DATASET_ROOT_DIR`.

```
<$DATASET_ROOT_DIR>
├── breeds
│   └── ILSVRC ⇒ ../imagenet/ILSVRC
├── imagenet
│   ├── ILSVRC
├── cifar10
├── cifar100
├── corrupt_cifar10
├── corrupt_cifar100
├── svhn
├── tinyimagenet
├── tinyimagenet_resize
├── wilds_animals
│   └── iwildcam_v2.0
└── wilds_camelyon
    └── camelyon17_v1.0
```

### Training

To get a list of all fully qualified names for all experiments in the paper, use

```bash
fd_shifts list
```

You can reproduce the results of the paper either all at once:

```bash
fd_shifts launch
```

Some at a time:

```bash
fd_shifts launch --model=devries --dataset=cifar10
```

Or one at a time (use `fd_shifts list` to find the names of experiments):

```bash
fd_shifts launch --name=fd-shifts/svhn_paper_sweep/devries_bbsvhn_small_conv_do1_run1_rew2.2
```

Check out `fd_shifts launch --help` for more filtering options.

### Inference

To run inference for one of the experiments, append `--mode=test` to any of the
commands above.

### Analysis

To run analysis for some of the predefined experiments, set `--mode=analysis` in
any of the commands above.

To run analysis over an already available set of model outputs the outputs have
to be in the following format:

For a classifier with `d` outputs and `N` samples in total (over all tested
datasets) and for `M` dropout samples

```
raw_logits.npz
Nx(d+2)

  0, 1, ...                 d─1,   d,      d+1
┌───────────────────────────────┬───────┬─────────────┐
|           logits_1            | label | dataset_idx |
├───────────────────────────────┼───────┼─────────────┤
|           logits_2            | label | dataset_idx |
├───────────────────────────────┼───────┼─────────────┤
|           logits_3            | label | dataset_idx |
└───────────────────────────────┴───────┴─────────────┘
.
.
.
┌───────────────────────────────┬───────┬─────────────┐
|           logits_N            | label | dataset_idx |
└───────────────────────────────┴───────┴─────────────┘
```

```
external_confids.npz
Nx1
```

```
raw_logits_dist.npz
NxdxM

  0, 1, ...                  d─1
┌───────────────────────────────┐
|   logits_1 (Dropout Sample 1) |
|   logits_1 (Dropout Sample 2) |
|               .               |
|               .               |
|               .               |
|   logits_1 (Dropout Sample M) |
├───────────────────────────────┤
|   logits_2 (Dropout Sample 1) |
|   logits_2 (Dropout Sample 2) |
|               .               |
|               .               |
|               .               |
|   logits_2 (Dropout Sample M) |
├───────────────────────────────┤
|   logits_3 (Dropout Sample 1) |
|   logits_3 (Dropout Sample 2) |
|               .               |
|               .               |
|               .               |
|   logits_3 (Dropout Sample M) |
└───────────────────────────────┘
                .
                .
                .
┌───────────────────────────────┐
|   logits_N (Dropout Sample 1) |
|   logits_N (Dropout Sample 2) |
|               .               |
|               .               |
|               .               |
|   logits_N (Dropout Sample M) |
└───────────────────────────────┘
```

```
external_confids_dist.npz
NxM
```

You may also use the `ExperimentData` class to load your data in another way.
You also have to provide an adequate config, where all test datasets and query
parameters are set. Check out the config files in `fd_shifts/configs` including
the dataclasses. Importantly, the `dataset_idx` has to match up with the list of
datasets you provide and whether or not `val_tuning` is set. If `val_tuning` is
set, the validation set takes over `dataset_idx=0`.

## Acknowledgements
