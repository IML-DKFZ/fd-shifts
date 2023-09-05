import argparse
import os
from pathlib import Path
from typing import Callable

import imageio.core.util
import matplotlib.pyplot as plt
import medpy.io as mpy
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io


def prepare_rxrx1(data_dir: Path):
    base_path = data_dir / "rxrx1"

    fl_micro_df = pd.read_csv(base_path / "metadata.csv")
    fl_micro_df["filepath"] = "NA"
    fl_micro_df["stempath"] = "NA"
    for iloc in range(len(fl_micro_df)):
        path = fl_micro_df["site_id"].iloc[iloc]
        experiment, plate_nb, well_id, sampleofwell = path.split("_")
        stempath = (
            f"images/{experiment}/Plate{plate_nb}/{well_id}_s{sampleofwell}_wXXX.png"
        )
        start, end = stempath.split("XXX")
        filepath = start + "RGB" + end
        print(stempath)
        print(filepath)
        fl_micro_df.iloc[iloc, fl_micro_df.columns.get_loc("filepath")] = filepath
        fl_micro_df.iloc[iloc, fl_micro_df.columns.get_loc("stempath")] = stempath

    fl_micro_df["target"] = fl_micro_df["sirna_id"]

    fl_micro_df.to_csv(base_path / "rxrx1_multiclass.csv")

    # Creating train and test for all imgs
    # Carefull: Not imgs from a well in both train and testset!

    df_all = pd.read_csv(base_path / "rxrx1_multiclass.csv")

    unique_wells = len(df_all.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(unique_wells, 12551, replace=False)

    id_test = df_all["well_id"].unique()[indices]

    df_test = df_all[df_all["well_id"].isin(id_test)]
    df_train = df_all[~df_all["well_id"].isin(id_test)]

    df_train.to_csv(base_path / "rxrx1_multiclass_all_train.csv")
    df_test.to_csv(base_path / "rxrx1_multiclass_all_test.csv")

    df_not_HEPG2 = df_all[~(df_all["cell_type"] == "HEPG2")]
    df_HEPG2 = df_all[(df_all["cell_type"] == "HEPG2")]

    df_not_HUVEC = df_all[~(df_all["cell_type"] == "HUVEC")]
    df_HUVEC = df_all[(df_all["cell_type"] == "HUVEC")]

    df_not_U2OS = df_all[~(df_all["cell_type"] == "U2OS")]
    df_U2OS = df_all[(df_all["cell_type"] == "U2OS")]

    df_not_RPE = df_all[~(df_all["cell_type"] == "RPE")]
    df_RPE = df_all[(df_all["cell_type"] == "RPE")]

    unique_wells_HEPG2 = len(df_not_HEPG2.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_HEPG2, int(unique_wells_HEPG2 * 0.2), replace=False
    )
    id_test = df_not_HEPG2["well_id"].unique()[indices]
    df_test_not_HEPG2 = df_not_HEPG2[df_not_HEPG2["well_id"].isin(id_test)]
    df_train_not_HEPG2 = df_not_HEPG2[~df_not_HEPG2["well_id"].isin(id_test)]

    unique_wells_HUVEC = len(df_not_HUVEC.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_HUVEC, int(unique_wells_HUVEC * 0.2), replace=False
    )
    id_test = df_not_HUVEC["well_id"].unique()[indices]
    df_test_not_HUVEC = df_not_HUVEC[df_not_HUVEC["well_id"].isin(id_test)]
    df_train_not_HUVEC = df_not_HUVEC[~df_not_HUVEC["well_id"].isin(id_test)]

    unique_wells_U2OS = len(df_not_U2OS.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_U2OS, int(unique_wells_U2OS * 0.2), replace=False
    )
    id_test = df_not_U2OS["well_id"].unique()[indices]
    df_test_not_U2OS = df_not_U2OS[df_not_U2OS["well_id"].isin(id_test)]
    df_train_not_U2OS = df_not_U2OS[~df_not_U2OS["well_id"].isin(id_test)]

    unique_wells_RPE = len(df_not_RPE.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_RPE, int(unique_wells_RPE * 0.2), replace=False
    )
    id_test = df_not_RPE["well_id"].unique()[indices]
    df_test_not_RPE = df_not_RPE[df_not_RPE["well_id"].isin(id_test)]
    df_train_not_RPE = df_not_RPE[~df_not_RPE["well_id"].isin(id_test)]

    df_test_not_HEPG2.to_csv(base_path / "rxrx1_multiclass_but_hepg2_test.csv")
    df_train_not_HEPG2.to_csv(base_path / "rxrx1_multiclass_but_hepg2_train.csv")
    df_HEPG2.to_csv(base_path / "rxrx1_multiclass_only_hepg2_test.csv")

    df_test_not_HUVEC.to_csv(base_path / "rxrx1_multiclass_but_huvec_test.csv")
    df_train_not_HUVEC.to_csv(base_path / "rxrx1_multiclass_but_huvec_train.csv")
    df_HUVEC.to_csv(base_path / "rxrx1_multiclass_only_huvec_test.csv")

    df_test_not_U2OS.to_csv(base_path / "rxrx1_multiclass_but_u2os_test.csv")
    df_train_not_U2OS.to_csv(base_path / "rxrx1_multiclass_but_u2os_train.csv")
    df_U2OS.to_csv(base_path / "rxrx1_multiclass_only_u2os_test.csv")

    df_test_not_RPE.to_csv(base_path / "rxrx1_multiclass_but_rpe_test.csv")
    df_train_not_RPE.to_csv(base_path / "rxrx1_multiclass_but_rpe_train.csv")
    df_RPE.to_csv(base_path / "rxrx1_multiclass_only_rpe_test.csv")

    # ### 40 Experimental Conditions vs 11 Experimental Conditions
    # 41 vs 10
    # 41 vs 10
    # 41 vs 10
    # 41 vs 10

    indices = np.random.choice(51, 51, replace=False)

    exp_all = df_all.experiment.unique()

    set1 = exp_all[indices[:11]]
    set2 = exp_all[indices[11:21]]
    set3 = exp_all[indices[21:31]]
    set4 = exp_all[indices[31:41]]
    set5 = exp_all[indices[41:]]

    df_set1_large = df_all[~df_all.experiment.isin(set1)]
    df_set2_large = df_all[~df_all.experiment.isin(set2)]
    df_set3_large = df_all[~df_all.experiment.isin(set3)]
    df_set4_large = df_all[~df_all.experiment.isin(set4)]
    df_set5_large = df_all[~df_all.experiment.isin(set5)]

    df_set1_small = df_all[df_all.experiment.isin(set1)]
    df_set2_small = df_all[df_all.experiment.isin(set2)]
    df_set3_small = df_all[df_all.experiment.isin(set3)]
    df_set4_small = df_all[df_all.experiment.isin(set4)]
    df_set5_small = df_all[df_all.experiment.isin(set5)]

    unique_wells_set1 = len(df_set1_large.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_set1, int(unique_wells_set1 * 0.2), replace=False
    )
    id_test = df_set1_large["well_id"].unique()[indices]
    df_set1_large_test = df_set1_large[df_set1_large["well_id"].isin(id_test)]
    df_set1_large_train = df_set1_large[~df_set1_large["well_id"].isin(id_test)]

    unique_wells_set2 = len(df_set2_large.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_set2, int(unique_wells_set2 * 0.2), replace=False
    )
    id_test = df_set2_large["well_id"].unique()[indices]
    df_set2_large_test = df_set2_large[df_set2_large["well_id"].isin(id_test)]
    df_set2_large_train = df_set2_large[~df_set2_large["well_id"].isin(id_test)]

    unique_wells_set3 = len(df_set3_large.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_set3, int(unique_wells_set3 * 0.2), replace=False
    )
    id_test = df_set3_large["well_id"].unique()[indices]
    df_set3_large_test = df_set3_large[df_set3_large["well_id"].isin(id_test)]
    df_set3_large_train = df_set3_large[~df_set3_large["well_id"].isin(id_test)]

    unique_wells_set4 = len(df_set4_large.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_set4, int(unique_wells_set4 * 0.2), replace=False
    )
    id_test = df_set4_large["well_id"].unique()[indices]
    df_set4_large_test = df_set4_large[df_set4_large["well_id"].isin(id_test)]
    df_set4_large_train = df_set4_large[~df_set4_large["well_id"].isin(id_test)]

    unique_wells_set5 = len(df_set5_large.well_id.unique())
    np.random.seed(2)
    indices = np.random.choice(
        unique_wells_set5, int(unique_wells_set5 * 0.2), replace=False
    )
    id_test = df_set5_large["well_id"].unique()[indices]
    df_set5_large_test = df_set5_large[df_set5_large["well_id"].isin(id_test)]
    df_set5_large_train = df_set5_large[~df_set5_large["well_id"].isin(id_test)]

    df_set1_large_test.to_csv(base_path / "rxrx1_multiclass_large_set1_test.csv")
    df_set1_large_train.to_csv(base_path / "rxrx1_multiclass_large_set1_train.csv")
    df_set1_small.to_csv(base_path / "rxrx1_multiclass_small_set1_test.csv")

    df_set2_large_test.to_csv(base_path / "rxrx1_multiclass_large_set2_test.csv")
    df_set2_large_train.to_csv(base_path / "rxrx1_multiclass_large_set2_train.csv")
    df_set2_small.to_csv(base_path / "rxrx1_multiclass_small_set2_test.csv")

    df_set3_large_test.to_csv(base_path / "rxrx1_multiclass_large_set3_test.csv")
    df_set3_large_train.to_csv(base_path / "rxrx1_multiclass_large_set3_train.csv")
    df_set3_small.to_csv(base_path / "rxrx1_multiclass_small_set3_test.csv")

    df_set4_large_test.to_csv(base_path / "rxrx1_multiclass_large_set4_test.csv")
    df_set4_large_train.to_csv(base_path / "rxrx1_multiclass_large_set4_train.csv")
    df_set4_small.to_csv(base_path / "rxrx1_multiclass_small_set4_test.csv")

    df_set5_large_test.to_csv(base_path / "rxrx1_multiclass_large_set5_test.csv")
    df_set5_large_train.to_csv(base_path / "rxrx1_multiclass_large_set5_train.csv")
    df_set5_small.to_csv(base_path / "rxrx1_multiclass_small_set5_test.csv")
