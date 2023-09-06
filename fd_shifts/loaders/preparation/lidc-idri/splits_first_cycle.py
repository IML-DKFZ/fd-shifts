import os
import pickle
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import pytorch_lightning as pl


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--feature",
        "-f",
        type=str,
        help="Metadata feature that is used to create the splits. We used texture and malignancy in our experiments.",
        default="texture",
    )
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Path to the cropped LIDC-IDRI dataset. If not given, the arguments --splits_path and --id_ood_csv "
        "have to be specified. If given, should contain the id_ood.csv file with the i.i.d. and OoD information "
        "about the nodules.",
        default=None,
    )
    parser.add_argument(
        "--splits_path",
        "-s",
        type=str,
        help="Path to store the created splits file. If not given, the argument --dataset_path has to be specified."
        "If given as directory, a file named splits.pkl will be created, otherwise has to be specified as .pkl file",
        default=None,
    )
    parser.add_argument(
        "--id_ood_csv",
        type=str,
        help="Path where the i.i.d. and OoD of the nodules is stored. "
        "If not given, the argument --dataset_path has to be specified.",
        default=None,
    )
    args = parser.parse_args()
    return args


def create_splits(output_path, shift_feature, metadata_csv, seed, n_splits=5) -> None:
    """Saves a pickle file containing the splits for k-fold cv on the dataset

    Args:
        output_path: The output path where to save the splits file
        shift_feature: The metadata feature which is used as a basis for the splits
        metadata_csv: The csv file with the i.i.d / OoD information of the nodules
        seed: The seed for the splits
        n_splits: Number of folds
    """
    np.random.seed(seed)
    # array which contains all the splits, one dictionary for each fold
    splits = []

    # read csv with metadata information about features
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df["Segmentation Save Paths"] = metadata_df[
        "Segmentation Save Paths"
    ].apply(
        lambda paths: [f"{path.split('/')[-1].split('.')[0]}.npy" for path in paths]
    )
    metadata_df["Image Save Path"] = metadata_df["Image Save Path"].apply(
        lambda path: f"{path.split('/')[-1].split('.')[0]}.npy"
    )
    # shift feature is given with "_" as separator in params,
    # while it is separated with space in the pandas dataframe
    # (only relevant for internal_Structure vs. internalStructure)
    # -> needs to be considered when accessing column in df
    shift_feature_id = f"{' '.join(shift_feature.split('_'))}_id"
    if shift_feature is None:
        id_train_patients = set(metadata_df["Patient ID"].unique())
        ood_patients = set()
    else:
        # get OOD images
        ood_patients = set()
        for index, row in metadata_df.iterrows():
            if row[shift_feature_id] == False:
                ood_patients.add(row["Patient ID"])
        id_train_patients = set()
        for index, row in metadata_df.iterrows():
            if row["Patient ID"] not in ood_patients and row[shift_feature_id] == True:
                id_train_patients.add(row["Patient ID"])

    ood_nodules = metadata_df.loc[
        (metadata_df["Patient ID"].isin(ood_patients))
        & (metadata_df[shift_feature_id] == False)
    ]
    num_ood_nodules = len(ood_nodules.index)

    num_unlabeled_pool = num_ood_nodules // 2
    ood_unlabeled_pool_patients = set()
    ood_unlabeled_pool = []
    id_unlabeled_pool = []

    while len(ood_unlabeled_pool) < num_unlabeled_pool:
        patient_to_add_unlabeled_pool = random.choice(sorted(list(ood_patients)))
        ood_unlabeled_pool_patients.add(patient_to_add_unlabeled_pool)
        ood_patients.remove(patient_to_add_unlabeled_pool)
        ood_unlabeled_pool.extend(
            metadata_df.loc[
                (metadata_df["Patient ID"] == patient_to_add_unlabeled_pool)
                & (metadata_df[shift_feature_id] == False),
                "Image Save Path",
            ].tolist()
        )
        id_unlabeled_pool.extend(
            metadata_df.loc[
                (metadata_df["Patient ID"] == patient_to_add_unlabeled_pool)
                & (metadata_df[shift_feature_id] == True),
                "Image Save Path",
            ].tolist()
        )

    ood_test = metadata_df.loc[
        (metadata_df["Patient ID"].isin(ood_patients))
        & (metadata_df[shift_feature_id] == False)
    ]["Image Save Path"].tolist()
    id_test = metadata_df.loc[
        (metadata_df["Patient ID"].isin(ood_patients))
        & (metadata_df[shift_feature_id] == True)
    ]["Image Save Path"].tolist()
    id_train = metadata_df.loc[
        (metadata_df["Patient ID"].isin(id_train_patients))
        & (metadata_df[shift_feature_id] == True)
    ]["Image Save Path"].tolist()

    all_id_cases = len(id_train) + len(id_test)
    num_id_train = int(0.8 * all_id_cases)
    num_id_test = all_id_cases - num_id_train
    id_test_patients = set()
    num_to_add = num_id_test - len(id_test)

    nodules_to_add_test = []
    while len(nodules_to_add_test) < num_to_add:
        patient_to_add_test = random.choice(sorted(list(id_train_patients)))
        id_test_patients.add(patient_to_add_test)
        id_train_patients.remove(patient_to_add_test)
        nodules_to_add_test.extend(
            metadata_df.loc[
                (metadata_df["Patient ID"] == patient_to_add_test)
                & (
                    metadata_df["{}_id".format(" ".join(shift_feature.split("_")))]
                    == True
                ),
                "Image Save Path",
            ].tolist()
        )

    id_test.extend(nodules_to_add_test)

    num_id_unlabeled_pool = len(ood_unlabeled_pool) * 2
    num_to_add = num_id_unlabeled_pool - len(id_unlabeled_pool)
    id_unlabeled_pool_patients = set()
    nodules_to_add_unlabeled_pool = []

    while len(nodules_to_add_unlabeled_pool) < num_to_add:
        patient_to_add_unlabeled_pool = random.choice(sorted(list(id_train_patients)))
        id_unlabeled_pool_patients.add(patient_to_add_unlabeled_pool)
        id_train_patients.remove(patient_to_add_unlabeled_pool)
        nodules_to_add_unlabeled_pool.extend(
            metadata_df.loc[
                (metadata_df["Patient ID"] == patient_to_add_unlabeled_pool)
                & (
                    metadata_df["{}_id".format(" ".join(shift_feature.split("_")))]
                    == True
                ),
                "Image Save Path",
            ].tolist()
        )
    id_unlabeled_pool.extend(nodules_to_add_unlabeled_pool)

    id_train = [
        path
        for path in id_train
        if path not in nodules_to_add_test and path not in nodules_to_add_unlabeled_pool
    ]

    assert len(id_train_patients) + len(ood_patients) + len(id_test_patients) == len(
        id_train_patients.union(ood_patients).union(id_test_patients)
    )

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # create fold dictionary and append it to splits
    for i, (train_idx, val_idx) in enumerate(kfold.split(id_train)):
        train_keys = np.array(id_train)[train_idx]
        val_keys = np.array(id_train)[val_idx]
        split_dict = dict()
        split_dict["train"] = train_keys
        split_dict["val"] = val_keys
        split_dict["id_test"] = id_test
        split_dict["ood_test"] = np.array(ood_test)
        split_dict["id_unlabeled_pool"] = np.array(id_unlabeled_pool)
        split_dict["ood_unlabeled_pool"] = np.array(ood_unlabeled_pool)
        splits.append(split_dict)

    with open(output_path, "wb") as f:
        pickle.dump(splits, f)


def main(args: Namespace):
    seed = 123
    pl.seed_everything(seed)
    feature = args.feature
    if args.dataset_path is None:
        if args.id_ood_csv is None:
            print(
                "If you didn't specify the dataset path with the cropped nodules, "
                "you need to specify where the i.i.d. and OoD information about the nodules is stored!"
            )
            return
        if args.splits_path is None:
            print(
                "If you didn't specify the dataset path with the cropped nodules, "
                "you need to specify where to store the splits file!"
            )
            return
    if args.dataset_path is not None:
        dataset_path = Path(args.dataset_path)
        id_ood_csv = dataset_path / "id_ood.csv"
        splits_path = dataset_path / "splits" / feature / "firstCycle" / "splits.pkl"
    if args.id_ood_csv is not None:
        id_ood_csv = Path(args.id_ood_csv)
    if args.splits_path is not None:
        splits_path = Path(args.splits_path)
        if not str(splits_path).endswith(".pkl"):
            splits_path = splits_path / "splits.pkl"
    os.makedirs(splits_path.parent, exist_ok=True)
    create_splits(
        output_path=splits_path,
        shift_feature=feature,
        metadata_csv=id_ood_csv,
        seed=seed,
    )


if __name__ == "__main__":
    cli_args = main_cli()
    main(args=cli_args)
