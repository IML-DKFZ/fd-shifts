# Reproducing ["Overcoming Common Flaws in the Evaluation of Selective Classification Systems"]()
For installation and general usage, follow the [FD-Shifts instructions](../../README.md).

## Data Folder Requirements

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

For information regarding where to download these datasets from and what you have to do with them please check out the [dataset documentation](../datasets.md).

## Training & Analysis

To get a list of all fully qualified names for all experiments in the paper, use

```bash
fd-shifts list-experiments --custom-filter=augrc2024
```

To reproduce all results of the paper:

```bash
fd-shifts launch --mode=train --custom-filter=augrc2024
fd-shifts launch --mode=test --custom-filter=augrc2024
fd-shifts launch --mode=analysis --custom-filter=augrc2024
```

```bash
python scripts/analysis_bootstrap.py --custom-filter=augrc2024
```

### Model Weights

All pretrained model weights used for the benchmark can be found on Zenodo under the following links:

- [iWildCam-2020-Wilds](https://zenodo.org/record/7620946)
- [iWildCam-2020-Wilds (OpenSet Training)](https://zenodo.org/record/7621150)
- [BREEDS-ENTITY-13](https://zenodo.org/record/7621249)
- [CAMELYON-17-Wilds](https://zenodo.org/record/7621456)
- [CIFAR-100](https://zenodo.org/record/7622086)
- [CIFAR-100 (superclasses)](https://zenodo.org/record/7622116)
- [CIFAR-10](https://zenodo.org/record/7622047)
- [SVHN](https://zenodo.org/record/7622152)
- [SVHN (OpenSet Training)](https://zenodo.org/record/7622177)

### Create results tables

```bash
fd-shifts report
fd-shifts report_bootstrap
```
