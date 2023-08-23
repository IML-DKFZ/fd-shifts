import logging
import os
import pickle
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

import fd_shifts.configs.data as data_configs
from fd_shifts import configs
from fd_shifts.loaders.dataset_collection import get_dataset
from fd_shifts.utils.aug_utils import get_transform, target_transforms_collection


class FDShiftsDataLoader(pl.LightningDataModule):
    """Data module class for combination of multiple datasets for testing with shifts"""

    def __init__(self, cf: configs.Config, no_norm_flag=False):
        super().__init__()
        self.crossval_ids_path = cf.exp.crossval_ids_path
        self.crossval_n_folds = cf.exp.crossval_n_folds
        self.fold = cf.exp.fold
        self.data_dir = cf.data.data_dir
        self.data_root_dir = cf.exp.data_root_dir
        self.dataset_name = cf.data.dataset
        self.batch_size = cf.trainer.batch_size
        self.pin_memory = cf.data.pin_memory
        self.num_workers = cf.data.num_workers
        self.reproduce_confidnet_splits = cf.data.reproduce_confidnet_splits
        self.dataset_kwargs = cf.data.kwargs
        self.devries_repro_ood_split = cf.test.devries_repro_ood_split
        self.val_split = cf.trainer.val_split.name
        self.test_iid_split = cf.test.iid_set_split
        self.assim_ood_norm_flag = cf.test.assim_ood_norm_flag
        self.balanced_sampeling = cf.model.balanced_sampeling
        self.add_val_tuning = cf.eval.val_tuning
        self.query_studies = dict(cf.eval.query_studies)
        if self.query_studies is not None:
            self.external_test_sets = []
            for key, values in self.query_studies.items():
                if key != "iid_study" and values is not None:
                    self.external_test_sets.extend(list(values))
            logging.debug(
                "CHECK flat list of external datasets %s", self.external_test_sets
            )

            if len(self.external_test_sets) > 0:
                self.external_test_configs = {}
                for ext_set in self.external_test_sets:
                    overwrite_dataset = False
                    if ext_set.startswith("dermoscopyall"):
                        file_set = "dermoscopyall"
                        overwrite_dataset = False
                    elif ext_set.startswith("rxrx1all"):
                        file_set = "rxrx1all"
                        overwrite_dataset = False
                    elif ext_set.startswith("lidc_idriall"):
                        file_set = "lidc_idriall"
                        overwrite_dataset = False
                    elif ext_set.startswith("xray_chestall"):
                        file_set = "xray_chestall"
                        overwrite_dataset = False
                    else:
                        file_set = ext_set
                    self.external_test_configs[ext_set] = OmegaConf.load(
                        os.path.join(
                            os.path.abspath(os.path.dirname(data_configs.__file__)),
                            "{}_data.yaml".format(file_set),
                        )
                    ).data
                    if overwrite_dataset:
                        self.external_test_configs[ext_set].dataset = ext_set
        # set up target transforms
        self.target_transforms = {}
        if cf.data.target_transforms:
            self.add_target_transforms(cf.data.target_transforms, no_norm_flag)

        # Set up augmentations
        self.augmentations = {}
        if cf.data.augmentations:
            self.add_augmentations(
                cf.data.augmentations,
                no_norm_flag,
            )

        self.train_dataset, self.val_dataset, self.test_datasets = None, None, None

    def add_target_transforms(self, query_tt, no_norm_flag):
        # add if for empty target transform. currently bug for no tt
        for datasplit_k, datasplit_v in query_tt.items():
            target_transforms, target_transforms_after = [], []
            if datasplit_v is not None:
                for tt_key, tt_param in datasplit_v.items():
                    target_transforms.append(
                        target_transforms_collection[tt_key](tt_param)
                    )
            self.target_transforms[datasplit_k] = target_transforms[0]
        print(
            "CHECK TARGET TRANSFORMS", self.assim_ood_norm_flag, self.target_transforms
        )

    def prepare_data(self, *args, **kwargs):
        pass

    def add_augmentations(self, query_augs, no_norm_flag):
        if self.query_studies is not None and self.external_test_sets is not None:
            for ext_set in self.external_test_sets:
                query_augs["external_{}".format(ext_set)] = self.external_test_configs[
                    ext_set
                ].augmentations["test"]
        for datasplit_k, datasplit_v in query_augs.items():
            augmentations, aug_after = [], []
            if datasplit_v is not None:
                for aug_key, aug_param in datasplit_v.items():
                    if aug_key == "normalize" and no_norm_flag is True:
                        pass
                    elif (
                        "external" in datasplit_k
                        and aug_key == "normalize"
                        and self.assim_ood_norm_flag
                    ):
                        logging.debug(
                            "assimilating norm of ood dataset to iid test set..."
                        )
                        aug_param = query_augs["test"]["normalize"]
                        augmentations.append(get_transform(aug_key, aug_param))

                    else:
                        augmentations.append(get_transform(aug_key, aug_param))
            self.augmentations[datasplit_k] = get_transform("compose", augmentations)
        logging.debug(
            "CHECK AUGMETNATIONS %s, %s", self.assim_ood_norm_flag, self.augmentations
        )

    def setup(self, stage=None):
        self.train_dataset = get_dataset(
            name=self.dataset_name,
            root=self.data_dir,
            train=True,
            download=True,
            target_transform=self.target_transforms.get("train"),
            transform=self.augmentations["train"],
            kwargs=self.dataset_kwargs,
        )
        logging.debug("Len Training data: %s", len(self.train_dataset))

        self.iid_test_set = get_dataset(
            name=self.dataset_name,
            root=self.data_dir,
            train=False,
            download=True,
            target_transform=self.target_transforms.get("test"),
            transform=self.augmentations["test"],
            kwargs=self.dataset_kwargs,
        )

        if self.test_iid_split == "tenPercent":
            length_test = len(self.iid_test_set)
            split = int(length_test * 0.1)
            if "wilds" in self.dataset_name:
                self.iid_test_set.indices = self.iid_test_set.indices[split:]
                self.iid_test_set.__len__ = len(self.iid_test_set.indices)
            else:
                try:
                    self.iid_test_set.imgs = self.iid_test_set.imgs[split:]
                    self.iid_test_set.samples = self.iid_test_set.samples[split:]
                    self.iid_test_set.targets = self.iid_test_set.targets[split:]
                    self.iid_test_set.__len__ = len(self.iid_test_set.imgs)
                except:
                    self.iid_test_set.data = self.iid_test_set.data[split:]
                    try:
                        self.iid_test_set.targets = self.iid_test_set.targets[split:]
                    except:
                        self.iid_test_set.labels = self.iid_test_set.labels[split:]
                    self.iid_test_set.__len__ = len(self.iid_test_set.data)

        if self.val_split == "devries":
            self.val_dataset = get_dataset(
                name=self.dataset_name,
                root=self.data_dir,
                train=False,
                download=True,
                target_transform=self.target_transforms.get("val"),
                transform=self.augmentations["val"],
                kwargs=self.dataset_kwargs,
            )
            if "wilds" in self.dataset_name:
                self.iid_test_set.indices = self.iid_test_set.indices[100:150]
                self.iid_test_set.__len__ = len(self.iid_test_set.indices)
            else:
                try:
                    self.iid_test_set.imgs = self.iid_test_set.imgs[1000:]
                    self.iid_test_set.samples = self.iid_test_set.samples[1000:]
                    self.iid_test_set.targets = self.iid_test_set.targets[1000:]
                    self.iid_test_set.__len__ = len(self.iid_test_set.imgs)
                except:
                    self.iid_test_set.data = self.iid_test_set.data[1000:]
                    try:
                        self.iid_test_set.targets = self.iid_test_set.targets[1000:]
                    except:
                        self.iid_test_set.labels = self.iid_test_set.labels[1000:]
                    self.iid_test_set.__len__ = len(self.iid_test_set.data)
            if self.val_split == "devries":
                self.val_dataset = get_dataset(
                    name=self.dataset_name,
                    root=self.data_dir,
                    train=False,
                    download=True,
                    target_transform=self.target_transforms.get("val"),
                    transform=self.augmentations["val"],
                    kwargs=self.dataset_kwargs,
                )
                if "wilds" in self.dataset_name:
                    self.val_dataset.indices = self.val_dataset.indices[:1000]
                    self.val_dataset.__len__ = len(self.val_dataset.indices)
                else:
                    try:
                        self.val_dataset.imgs = self.val_dataset.imgs[:1000]
                        self.val_dataset.samples = self.val_dataset.samples[:1000]
                        self.val_dataset.targets = self.val_dataset.targets[:1000]
                        self.val_dataset.__len__ = len(self.val_dataset.imgs)
                    except:
                        self.val_dataset.data = self.val_dataset.data[:1000]
                        try:
                            self.val_dataset.targets = self.val_dataset.targets[:1000]
                        except:
                            self.val_dataset.labels = self.val_dataset.labels[:1000]
                        self.val_dataset.__len__ = len(self.val_dataset.data)

        else:
            self.val_dataset = get_dataset(
                name=self.dataset_name,
                root=self.data_dir,
                train=True,
                download=True,
                target_transform=self.target_transforms.get("val"),
                transform=self.augmentations["val"],
                kwargs=self.dataset_kwargs,
            )

        logging.debug("Len Val data: %s", len(self.val_dataset))
        logging.debug("Len iid test data: %s", len(self.iid_test_set))

        self.test_datasets = []

        if self.add_val_tuning:
            self.test_datasets.append(self.val_dataset)
            logging.debug(
                "Adding tuning data. (preliminary) len: %s", len(self.test_datasets[-1])
            )

        if not (
            self.query_studies is not None and "iid_study" not in self.query_studies
        ):
            self.test_datasets.append(self.iid_test_set)
            logging.debug(
                "Adding internal test dataset. %s", len(self.test_datasets[-1])
            )

        if self.query_studies is not None and len(self.external_test_sets) > 0:
            for ext_set in self.external_test_sets:
                logging.debug("Adding external test dataset: %s", ext_set)
                tmp_external_set = get_dataset(
                    name=ext_set,
                    root=os.path.join(
                        self.data_root_dir, self.external_test_configs[ext_set].dataset
                    ),
                    train=False,
                    download=True,
                    target_transform=self.target_transforms,
                    transform=self.augmentations["external_{}".format(ext_set)],
                    kwargs=self.dataset_kwargs,
                )
                if (
                    self.devries_repro_ood_split
                    and ext_set in self.query_studies["new_class_study"]
                ):
                    try:
                        tmp_external_set.imgs = tmp_external_set.imgs[1000:]
                        tmp_external_set.samples = tmp_external_set.samples[1000:]
                        tmp_external_set.__len__ = len(tmp_external_set.imgs)
                    except:
                        tmp_external_set.data = tmp_external_set.data[1000:]
                        tmp_external_set.__len__ = len(tmp_external_set.data)

                    logging.debug(
                        "shortened external set %s to len %s".format(
                            ext_set, len(tmp_external_set)
                        )
                    )
                self.test_datasets.append(tmp_external_set)
                logging.debug("Len external Test data: %s", len(self.test_datasets[-1]))

        if self.val_split is None or self.val_split == "devries":
            val_idx = []
            train_idx = []
            self.val_sampler = None
            self.train_sampler = None
            if self.balanced_sampeling:
                # do class balanced sampeling
                val_idx = []
                train_idx = []
                self.val_sampler = None
                class_weights = {}
                labels = self.train_dataset.csv.target

                for cla in labels.unique():
                    class_weights[cla] = 1 / (np.mean(labels == cla))

                sample_weights = [*map(class_weights.get, labels)]

                self.train_sampler = WeightedRandomSampler(
                    sample_weights, num_samples=len(sample_weights), replacement=True
                )

        elif self.val_split == "repro_confidnet":
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train))
            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            logging.debug(
                "reproduced train_val_splits from confidnet with val_idxs: %s",
                val_idx[:10],
            )
            self.val_sampler = val_idx
            self.train_sampler = SubsetRandomSampler(train_idx)

        elif self.val_split == "cv":
            if os.path.isfile(self.crossval_ids_path):
                with open(self.crossval_ids_path, "rb") as f:
                    train_idx, val_idx = pickle.load(f)[self.fold]
            else:
                num_train = len(self.train_dataset)
                indices = list(range(num_train))
                kf = KFold(n_splits=self.crossval_n_folds, shuffle=True, random_state=0)
                splits = list(kf.split(indices))
                train_idx, val_idx = splits[self.fold]
                self.val_sampler = val_idx

                with open(self.crossval_ids_path, "wb") as f:
                    pickle.dump(splits, f)
            self.train_sampler = SubsetRandomSampler(train_idx)

        else:
            raise NotImplementedError

        logging.debug("len train sampler %s", len(train_idx))
        logging.debug("len val sampler %s", len(val_idx))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=True if self.train_sampler is None else False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        if self.val_split == "zhang":
            val_loader = []
            for ix, test_dataset in enumerate(self.test_datasets[:2]):
                val_loader.append(
                    torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        pin_memory=self.pin_memory,
                        num_workers=self.num_workers,
                    )
                )
        else:
            val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,  # same dataset as train but potentially differing augs.
                batch_size=self.batch_size,
                sampler=self.val_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            )

        return val_loader

    def test_dataloader(
        self,
    ):
        test_loaders = []
        for ix, test_dataset in enumerate(self.test_datasets):
            test_loaders.append(
                torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                )
            )
        return test_loaders
