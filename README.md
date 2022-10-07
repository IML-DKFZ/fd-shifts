# FD-Shifts

![overview](./docs/new_overview.png)

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

```bash
pip install https://github.com/iml-dkfz/failure-detection-benchmark
```

If you have issues with cuda, install a recent PyTorch version according to their website.

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

For the predefined experiments we expect the data to be in the following folder structure relative
to the folder you set for `$DATASET_ROOT_DIR`.

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

To run inference for one of the experiments, append `--mode=test` to any of the commands above.

### Analysis

To run analysis for some of the predefined experiments, set `--mode=analysis` in any of the commands
above.

To run analysis over an already available set of model outputs the outputs have to be in the following format:

For a classifier with `d` outputs and `N` samples in total (over all tested datasets) and for `M`
dropout samples

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
parameters are set. Check out the config files in `fd_shifts/configs` including the dataclasses.
Importantly, the `dataset_idx` has to match up with the list of datasets you provide and whether or
not `val_tuning` is set. If `val_tuning` is set, the validation set takes over `dataset_idx=0`.

## Acknowledgements
