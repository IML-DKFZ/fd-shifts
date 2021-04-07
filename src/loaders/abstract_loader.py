import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.utils.aug_utils import transforms_collection
from sklearn.model_selection import KFold
import os
import pickle
import numpy as np


class AbstractDataLoader(pl.LightningDataModule):

    def __init__(self, cf):

        super().__init__()
        self.crossval_ids_path = cf.exp.crossval_ids_path
        self.crossval_n_folds = cf.exp.crossval_n_folds
        self.fold = cf.exp.fold
        self.data_dir = cf.data.data_dir
        self.batch_size = cf.trainer.batch_size
        self.pin_memory = cf.data.pin_memory
        self.num_workers = cf.data.num_workers
        self.reproduce_confidnet_splits = cf.data.reproduce_confidnet_splits

        # Set up augmentations
        self.augmentations = {}
        if cf.data.augmentations:
            self.add_augmentations(OmegaConf.to_container(cf.data.augmentations, resolve=True))

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None



    def add_augmentations(self, query_augs):

        for datasplit_k, datasplit_v in query_augs.items():
            augmentations, aug_after = [], []
            for aug_key, aug_param in datasplit_v.items():
                augmentations.append(transforms_collection[aug_key](aug_param))

            augmentations.append(transforms_collection["to_tensor"]())
            self.augmentations[datasplit_k] = transforms_collection["compose"](augmentations)


    def prepare_data(self):
        pass


    def setup(self, stage=None):

        if self.reproduce_confidnet_splits:
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train)) # they had valid_size at 0.1 in experiments
            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            print("reproduced train_val_splits from confidnet with val_idxs:", val_idx[:10])

        elif os.path.isfile(self.crossval_ids_path):
            with open(self.crossval_ids_path, "rb") as f:
                train_idx, val_idx = pickle.load(f)[self.fold]
        else:
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            kf = KFold(n_splits=self.crossval_n_folds, shuffle=True,random_state=0)
            splits = list(kf.split(indices))
            train_idx, val_idx = splits[self.fold]
            with open(self.crossval_ids_path, "wb") as f:
                pickle.dump(splits, f)


        # Make samplers
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = val_idx


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_dataset, #same dataset as train but potentially differing augs.
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            )


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

