# Datasets

These are instructions on how to get and modify the datasets used in the
benchmark, sorted by what dataset is used for training.

## Training on CIFAR-10, CIFAR-100, SVHN

The train and i.i.d. test datasets will automatically be downloaded.

### Corrupt CIFAR10, Corrupt CIFAR100

You can download CIFAR-10-C from [here](https://zenodo.org/record/2535967) and
CIFAR-100-C from [here](https://zenodo.org/record/3555552) using these (or similar) commands:
```bash
cd $DATASET_ROOT_DIR
mkdir corrupt_cifar10
cd corrupt_cifar10
wget https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
tar xf CIFAR-10-C.tar
mv CIFAR-10-C CIFAR-C
```

```bash
cd $DATASET_ROOT_DIR
mkdir corrupt_cifar100
cd corrupt_cifar100
wget https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
tar xf CIFAR-100-C.tar
mv CIFAR-100-C CIFAR-C
```

### tinyimagenet

You can download tinyimagenet from [the imagenet
website](https://image-net.org/download-images.php) after requesting access.
Preprocess the data with the following commands:

```bash
unzip tiny-imagenet-200.zip -d $DATASET_ROOT_DIR
cd $DATASET_ROOT_DIR
mv tiny-imagenet-200 tinyimagenet
cd tinyimagenet
rm -rf train test
cat val/val_annotations.txt | cut --output-delimiter=$'\n' -f 1,2 | xargs --verbose -n2 bash -c 'mkdir -p test/$1/images && mv val/images/$0 test/$1/images'
```

### tinyimagenet_resize

You can download the dataset from [here](https://github.com/ShiyuLiang/odin-pytorch) using these (or similar) commands:

```bash
cd $DATASET_ROOT_DIR
mkdir tinyimagenet_resize
cd tinyimagenet_resize
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
tar --strip-components=1 -xzf Imagenet_resize.tar.gz
```

## Training on BREEDS

You need to download the ImageNet ILSVRC2012 Task 1 & 2 dataset from [their
website](https://image-net.org/download-images.php) and move or symlink it to
`$DATASET_ROOT_DIR/breeds/ILSVRC`.

## Training on CAMELYON-17-Wilds, iWildCam-2020-Wilds

These datasets will be downloaded automatically.

## Dermoscopy Data

You can download the individual datasets from their respective websites:

- [PH2](https://www.fc.up.pt/addi/ph2%20database.html) (extract to `$DATASET_ROOT_DIR/ph2`)
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) (extract to `$DATASET_ROOT_DIR/ham10000`)
- [derm7pt](http://derm.cs.sfu.ca/) (extract to `$DATASET_ROOT_DIR/d7p`)
- [isic2020](https://challenge.isic-archive.com/data/#2020) (extract to `$DATASET_ROOT_DIR/isic_2020`)

Unpack them into their respective folders, then run the data preprocessing:

```bash
fd_shifts prepare --dataset dermoscopy
```

## Microscopy Data

Download the dataset from their website and extract to `$DATASET_ROOT_DIR/rxrx1`.

- [Rxrx1](https://www.rxrx.ai/rxrx1#Download)

Then run the data preprocessing:

```bash
fd_shifts prepare --dataset microscopy
```

## Chest XRay Data

You can download the individual datasets from their respective websites:

> **Note**  
> MIMIC requires you to apply for credentialed access.

- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) (extract to `$DATASET_ROOT_DIR/chexpert`)
- [MIMIC](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) (extract to `$DATASET_ROOT_DIR/mimic`)
- [NIH14](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) (extract to `$DATASET_ROOT_DIR/nih14`)

Unpack them into their respective folders, then run the data preprocessing:

```bash
fd_shifts prepare --dataset xray
```

## Lung CT Data

Download the dataset from their website and extract to `$DATASET_ROOT_DIR/lidc_idri`.

- [LIDC-IDRI](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)

Prepare the dataset by following the instructions in [the separate LIDC readme](../fd_shifts/loaders/preparation/lidc-idri/README.md).

Finaly, run the data preprocessing:

```bash
fd_shifts prepare --dataset lung_ct
```
