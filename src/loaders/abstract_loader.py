import torch
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.utils.aug_utils import transforms_collection
from src.loaders.dataset_collection import get_dataset
from sklearn.model_selection import KFold
import src.configs.data as data_configs
import os
import pickle
import numpy as np
from copy import deepcopy


class AbstractDataLoader(pl.LightningDataModule):

    def __init__(self, cf):

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
        self.dataset_kwargs = dict(cf.data).get("kwargs")
        self.devries_repro_ood_split = cf.test.devries_repro_ood_split
        self.val_split = cf.trainer.val_split
        self.test_iid_split = cf.test.iid_set_split
        self.assim_ood_norm_flag = cf.test.get("assim_ood_norm_flag")

        self.add_val_tuning = dict(cf.eval).get("val_tuning")
        self.query_studies = dict(cf.eval).get("query_studies")
        if self.query_studies is not None:
            self.external_test_sets = []
            for k in self.query_studies.keys():
                if k !="iid_study" and self.query_studies[k] is not None:
                    self.external_test_sets.extend([v for v in self.query_studies[k]])
            print("CHECK flat list of external datasets", self.external_test_sets)

            if len(self.external_test_sets) > 0:
                self.external_test_configs = {}
                for ext_set in self.external_test_sets:
                    self.external_test_configs[ext_set] = OmegaConf.load(os.path.join(os.path.abspath(os.path.dirname(data_configs.__file__)), "{}_data.yaml".format(ext_set)))

        # Set up augmentations
        self.augmentations = {}
        if cf.data.augmentations:
            self.add_augmentations(OmegaConf.to_container(cf.data.augmentations, resolve=True))

        self.train_dataset, self.val_dataset, self.test_datasets = None, None, None

    def add_augmentations(self, query_augs):

        if self.query_studies is not None and self.external_test_sets is not None:
            for ext_set in self.external_test_sets:
                query_augs["external_{}".format(ext_set)] = self.external_test_configs[ext_set].augmentations["test"]
        for datasplit_k, datasplit_v in query_augs.items():
            augmentations, aug_after = [], []
            if datasplit_v is not None:
                for aug_key, aug_param in datasplit_v.items():
                        if aug_key == "to_tensor":
                            augmentations.append(transforms_collection[aug_key])
                        elif "external" in datasplit_k and aug_key == "normalize" and self.assim_ood_norm_flag:
                            print("assimilating norm of ood dataset to iid test set...")
                            aug_param = query_augs["test"]["normalize"]
                            augmentations.append(transforms_collection[aug_key](aug_param))
                        else:
                            augmentations.append(transforms_collection[aug_key](aug_param))
            self.augmentations[datasplit_k] = transforms_collection["compose"](augmentations)
        print("CHECK AUGMETNATIONS", self.assim_ood_norm_flag, self.augmentations)


    def prepare_data(self):

        self.train_dataset = get_dataset(name=self.dataset_name,
                                         root=self.data_dir,
                                         train=True,
                                         download=True,
                                         transform=self.augmentations["train"],
                                         kwargs=self.dataset_kwargs
                                         )
        print("Len Training data: ", len(self.train_dataset))


        self.iid_test_set = get_dataset(name=self.dataset_name,
                                                  root=self.data_dir,
                                                  train=False,
                                                  download=True,
                                                  transform=self.augmentations["test"],
                                                  kwargs=self.dataset_kwargs
                                                  )

        if self.test_iid_split == "devries":
            try:
                self.iid_test_set.imgs = self.iid_test_set.imgs[1000:]
                self.iid_test_set.samples = self.iid_test_set.samples[1000:]
                self.iid_test_set.targets = self.iid_test_set.targets[1000:]
                self.iid_test_set.__len__ = len(self.iid_test_set.imgs)
            except:
                self.iid_test_set.data = self.iid_test_set.data[1000:]
                self.iid_test_set.targets = self.iid_test_set.targets[1000:]
                self.iid_test_set.__len__ = len(self.iid_test_set.data)

        if self.val_split == "devries":
            self.val_dataset = get_dataset(name=self.dataset_name,
                                           root=self.data_dir,
                                           train=False,
                                           download=True,
                                           transform=self.augmentations["val"],
                                           kwargs=self.dataset_kwargs
                                           )
            try:
                self.val_dataset.imgs = self.val_dataset.imgs[:1000]
                self.val_dataset.samples = self.val_dataset.samples[:1000]
                self.val_dataset.targets = self.val_dataset.targets[:1000]
                self.val_dataset.__len__ = len(self.val_dataset.imgs)
            except:
                self.val_dataset.data = self.val_dataset.data[:1000]
                self.val_dataset.targets = self.val_dataset.targets[:1000]
                self.val_dataset.__len__ = len(self.val_dataset.data)

        else:
            self.val_dataset = get_dataset(name=self.dataset_name,
                                           root=self.data_dir,
                                           train=True,
                                           download=True,
                                           transform=self.augmentations["val"],
                                           kwargs=self.dataset_kwargs
                                           )

        print("Len Val data: ", len(self.val_dataset))
        print("Len iid test data: ", len(self.iid_test_set))

        self.test_datasets = []

        if self.add_val_tuning:
            self.test_datasets.append(self.val_dataset)  # sampler defined later
            print("Adding tuning data. (preliminary) len: ", len(self.test_datasets[-1]))

        if not (self.query_studies is not None and "iid_study" not in self.query_studies):
            self.test_datasets.append(self.iid_test_set)
            print("Adding internal test dataset.", len(self.test_datasets[-1]))

        if self.query_studies is not None and len(self.external_test_sets) > 0:
            for ext_set in self.external_test_sets:
                print("Adding external test dataset:", ext_set)
                tmp_external_set = get_dataset(name=ext_set,
                                root=os.path.join(self.data_root_dir, ext_set),
                                train=False,
                                download=True,
                                transform=self.augmentations["external_{}".format(ext_set)],
                                kwargs=self.dataset_kwargs
                                )
                if self.devries_repro_ood_split and ext_set in self.query_studies["new_class_study"]:
                    try:
                        tmp_external_set.imgs = tmp_external_set.imgs[1000:]
                        tmp_external_set.samples = tmp_external_set.samples[1000:]
                        tmp_external_set.__len__ = len(tmp_external_set.imgs)
                    except:
                        tmp_external_set.data = tmp_external_set.data[1000:]
                        tmp_external_set.__len__ = len(tmp_external_set.data)

                    print("shortened external set {} to len {}".format(ext_set, len(tmp_external_set)))
                self.test_datasets.append(tmp_external_set)
                print("Len external Test data: ", len(self.test_datasets[-1]))



    def setup(self, stage=None):

        # val_split: None, repro_confidnet, devries, cv
        if self.val_split is None or self.val_split == "devries" or self.val_split == "zhang":
            num_train = len(self.train_dataset)
            train_idx = list(range(num_train))
            val_idx = []
            self.val_sampler = None


        elif self.val_split == "repro_confidnet":
            num_train = len(self.train_dataset)
            indices = list(range(num_train))
            split = int(np.floor(0.1 * num_train)) # they had valid_size at 0.1 in experiments
            np.random.seed(42)
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            print("reproduced train_val_splits from confidnet with val_idxs:", val_idx[:10])
            self.val_sampler = val_idx

        elif self.val_split == "cv":
            if os.path.isfile(self.crossval_ids_path):
                with open(self.crossval_ids_path, "rb") as f:
                    train_idx, val_idx = pickle.load(f)[self.fold]
            else:
                num_train = len(self.train_dataset)
                indices = list(range(num_train))
                kf = KFold(n_splits=self.crossval_n_folds, shuffle=True,random_state=0)
                splits = list(kf.split(indices))
                train_idx, val_idx = splits[self.fold]
                self.val_sampler = val_idx

                with open(self.crossval_ids_path, "wb") as f:
                    pickle.dump(splits, f)

        else:
            raise NotImplementedError


        # Make samplers
        self.train_sampler = SubsetRandomSampler(train_idx)

        print("len train sampler", len(self.train_sampler))
        print("len val sampler", len(val_idx))


    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            # shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=4,
            persistent_workers=True,
            )


    def val_dataloader(self):

        if self.val_split == "zhang":
            val_loader = []
            for ix, test_dataset in enumerate(self.test_datasets[:2]): # only iid test set and first ood set.
                val_loader.append(torch.utils.data.DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=self.pin_memory,
                    num_workers=self.num_workers,
                ))
        else:
            val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset, #same dataset as train but potentially differing augs.
                batch_size=self.batch_size,
                sampler=self.val_sampler,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                )

        return val_loader

    def test_dataloader(self):
        test_loaders = []
        for ix, test_dataset in enumerate(self.test_datasets):
            test_loaders.append(torch.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
            ))
        return test_loaders



