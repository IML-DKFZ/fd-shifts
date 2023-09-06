import ast
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        "-d",
        type=str,
        help="Path to the cropped LIDC-IDRI dataset. Should contain the metadata.csv file",
        required=True,
    )
    parser.add_argument(
        "--save_df",
        "-s",
        type=bool,
        help="(Should normally not be changed) Whether the analysis of id and ood should be stored as csv file.",
        default=True,
    )
    args = parser.parse_args()
    return args


def get_feature_dict():
    """
    Returns a dictionary with all the metadata features which can be analysed. The tuple for each feature determines
    which corresponding categories should be seen as i.i.d (first entry) and which as OoD (second entry)
    """
    return {
        "internal Structure": ((1,), (2, 3, 4)),
        "calcification": ((6,), (1, 2, 3, 4, 5)),
        "sphericity": ((3, 4, 5), (1, 2)),
        "lobulation": ((1, 2), (3, 4, 5)),
        "spiculation": ((1, 2), (3, 4, 5)),
        "texture": ((3, 4, 5), (1, 2)),
        "malignancy": ((1, 2, 3), (4, 5)),
    }


def calculate_rater_agreement(args: Namespace):
    dataset_path = Path(args.dataset_path)
    metadata_df = pd.read_csv(dataset_path / "metadata.csv")
    features_to_analyse = get_feature_dict()

    for feature, category in features_to_analyse.items():
        print(f"Feature: {feature}; ID: {category[0]}, OOD: {category[1]}")

    for column in metadata_df[features_to_analyse.keys()]:
        metadata_df[column] = metadata_df[column].apply(lambda x: ast.literal_eval(x))
        metadata_df[column] = metadata_df[column].apply(
            lambda ratings: None if "None" in str(ratings) else ratings
        )

        # Filter out columns with None entries, i.e. nodules with empty segmentation masks
        metadata_df = metadata_df[metadata_df[column].notnull()]
        # Binarize i.i.d and OoD categories
        metadata_df[column] = metadata_df[column].apply(
            lambda ratings: [
                1 if rating in features_to_analyse[column][0] else 0
                for rating in ratings
            ]
        )
        # Find nodules that have a majority vote, i.e. where not two raters voted i.i.d and two OoD
        metadata_df[f"{column}_majority"] = metadata_df[column].apply(
            lambda ratings: True if ratings.count(0) != ratings.count(1) else False
        )
        # Find i.i.d nodules
        metadata_df["{}_id".format(column)] = metadata_df[column].apply(
            lambda ratings: True if ratings.count(1) > ratings.count(0) else False
        )

        # Set all non-majority columns to None
        mask = metadata_df[f"{column}_majority"].tolist()
        mask = [not elem for elem in mask]
        metadata_df[f"{column}_id"][mask] = None
        metadata_df.drop([f"{column}_majority"], axis=1, inplace=True)
        print(metadata_df[f"{column}_id".format(column)].value_counts(dropna=False))
        print("=====================================================================")
        if args.save_df:
            metadata_df.to_csv(dataset_path / "id_ood.csv")


if __name__ == "__main__":
    cli_args = main_cli()
    calculate_rater_agreement(cli_args)
