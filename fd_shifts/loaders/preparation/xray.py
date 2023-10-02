import os
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_xray(data_dir: Path):
    train = pd.read_csv(data_dir / "chexpert/train.csv")

    val = pd.read_csv(data_dir / "chexpert/valid.csv")

    train = pd.concat([train, val])

    train_dropped = train.drop(
        [
            "Enlarged Cardiomediastinum",
            "Pleural Other",
            "Fracture",
            "Support Devices",
            "Lung Opacity",
            "Lung Lesion",
        ],
        axis=1,
    )

    td = train_dropped
    td["Consolidation"]

    pathologies = [
        "No Finding",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
    ]
    for path in pathologies:
        td[path] = (td[path] == 1).astype(int)

    td["Consolidation"]

    td = td[
        td[
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ].sum(axis=1)
        < 2
    ]

    # Create target column

    td["target"] = 0

    for loc in range(len(td)):
        for count, path in enumerate(
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ):
            if td.iloc[loc][path] == 1:
                td.iloc[loc, td.columns.get_loc("target")] = count

    td = td.rename(columns={"Path": "filepath"})

    for i in range(7):
        print(np.sum(td["target"] == i) * 100 / len(td))

    td.to_csv("cheXpert_multiclass")

    # ### NIH14 Dataset
    # Complexity here is that labels are in one column seperated by |

    nih14 = pd.read_csv(data_dir / "nih14/Data_Entry_2017.csv")

    paths = [
        "No Finding",
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    paths_com = [
        "No Finding",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Effusion",
    ]
    for path in paths:
        nih14[path] = 0

    for loc in range(len(nih14)):
        print(loc)
        labels = nih14["Finding Labels"].apply(lambda x: x.split("|")).iloc[loc]
        for label in labels:
            nih14.iloc[loc, nih14.columns.get_loc(label.strip())] = 1

    nih14d = nih14.drop(
        [
            "Infiltration",
            "Mass",
            "Nodule",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ],
        axis=1,
    )

    nih14d["Pleural Effusion"] = nih14d["Effusion"]
    nih14d = nih14d.drop(["Effusion"], axis=1)

    nih14d = nih14d[
        nih14d[
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ].sum(axis=1)
        == 1
    ]

    nih14d["target"] = 0

    for loc in range(len(nih14d)):
        for count, path in enumerate(
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ):
            if nih14d.iloc[loc][path] == 1:
                nih14d.iloc[loc, nih14d.columns.get_loc("target")] = count

    # class balance

    for i in range(7):
        print(sum(nih14d["target"] == i) * 100 / len(nih14d))

    imglist = []
    for i in range(12):
        if i + 1 < 10:
            nb = f"0{i+1}"
        else:
            nb = f"{i+1}"
        imgfolder = data_dir / f"nih14/images_0{nb}/images"
        print(f"{imgfolder=}")
        lst = os.listdir(imgfolder)
        lst.sort()
        imglist.append(lst[0])

    imgfolder = data_dir / "archive/images_001/images"
    lst = os.listdir(imgfolder)
    lst.sort()
    imglist.append(lst[0])

    for i in range(len(imglist)):
        if i + 1 < 10:
            nb = f"0{i+1}"
        else:
            nb = f"{i+1}"
        imgpath = f"images_0{nb}/images/"
        for loc in range(len(nih14d)):
            curimg = nih14d.iloc[loc, nih14d.columns.get_loc("Image Index")]
            if i < 11:
                if curimg >= imglist[i] and curimg < imglist[i + 1]:
                    nih14d.iloc[
                        loc, nih14d.columns.get_loc("Image Index")
                    ] = f"{imgpath}{curimg}"
                    print(f"{imgpath}{curimg}")
            elif i == 11:
                if curimg >= imglist[i] and curimg < "0028174_001.png":
                    nih14d.iloc[
                        loc, nih14d.columns.get_loc("Image Index")
                    ] = f"{imgpath}{curimg}"
                    print(f"{imgpath}{curimg}")

    nih14d = nih14d.rename(columns={"Image Index": "filepath"})

    nih14d["attribution"] = "nih14"
    nih14d.to_csv("nih14_multiclass")

    # # Mimic

    mimic = pd.read_csv(
        data_dir / "mimic/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
    )

    mimic_filepaths = pd.read_csv(
        data_dir / "mimic/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv"
    )

    mimic_filepaths["filepath"] = "missing"
    for i in range(len(mimic_filepaths)):
        mimic_filepaths.iloc[i]
        root = "files/mimic-cxr-jpg/2.0.0/files"
        subject_id = str(mimic_filepaths.subject_id.iloc[i])
        study_id = str(mimic_filepaths.study_id.iloc[i])
        img = mimic_filepaths.dicom_id.iloc[i]
        img_folder = f"{root}/p{subject_id[0:2]}/p{subject_id}/s{study_id}/{img}.jpg"
        mimic_filepaths.iloc[
            i, mimic_filepaths.columns.get_loc("filepath")
        ] = img_folder
        if i % 10000 == 0:
            print(i)

    mimic_df = pd.merge(mimic, mimic_filepaths, how="outer")

    pathologies = [
        "No Finding",
        "Cardiomegaly",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
    ]
    for path in pathologies:
        mimic_df[path] = (mimic_df[path] == 1).astype(int)

    mimic_df_droped = mimic_df[
        mimic_df[
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ].sum(axis=1)
        == 1
    ]
    # there is a bug. Since some labels are marked with -1 the sum can become 1
    # with there is one more positve finding than uncertain findings thus in
    # the final mimic csv there is a small amount of images with multiple
    # positve labels but only one is labeld. This leads to a drop in classifier
    # performance, since a correct prediction for one of the true classes is
    # only accepted as correct in 1/n_true.

    mimic_df["target"] = 0

    for loc in range(len(mimic_df)):
        for count, path in enumerate(
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ):
            if mimic_df.iloc[loc][path] == 1:
                mimic_df.iloc[loc, mimic_df.columns.get_loc("target")] = count

    mimic_df = mimic_df[
        mimic_df[
            [
                "No Finding",
                "Cardiomegaly",
                "Edema",
                "Consolidation",
                "Pneumonia",
                "Atelectasis",
                "Pneumothorax",
                "Pleural Effusion",
            ]
        ].sum(axis=1)
        == 1
    ]

    mimic_df.to_csv("mimic_multiclass.csv")

    mimic_df = pd.read_csv("mimic_multiclass.csv")
    nih_df = pd.read_csv("nih14_multiclass")
    chex_df = pd.read_csv("chexpert_multiclass")

    # ## Create shifts
    # 1) leave out
    # 1.1) Mimic
    # 1.2) Chexpert
    # 1.3) nih14

    mimic_df["patient_id"] = "mimic" + mimic_df["subject_id"].astype(str)
    mimic_df["attribution"] = "mimic"
    mimic_small = mimic_df[["patient_id", "target", "attribution", "filepath"]]
    mimic_small.to_csv("mimic_small.csv")

    nih14d["patient_id"] = nih14d["Patient ID"]
    td["patient_id"] = "missing"
    for i in range(len(td)):
        td.iloc[i, td.columns.get_loc("patient_id")] = td.filepath.iloc[i].split("/")[2]

    nih14d["attribution"] = "nih14"
    td["attribution"] = "chexpert"

    nih14_small = nih14d[["patient_id", "target", "attribution", "filepath"]]
    chexpert_small = td[["patient_id", "target", "attribution", "filepath"]]

    nih14_small.to_csv("nih14_small.csv")
    chexpert_small.to_csv("chexpert_small.csv")

    mimic_small = pd.read_csv("mimic_small.csv")
    nih14_small = pd.read_csv("nih14_small.csv")
    chexpert_small = pd.read_csv("chexpert_small.csv")

    chexpert_fp = chexpert_small.filepath.str.split("CheXpert-v1.0-small/", expand=True)
    chexpert_small["filepath"] = chexpert_fp[1]
    df_all = pd.concat([mimic_small, nih14_small, chexpert_small])

    patients_ls = df_all["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.98), replace=False
    )
    id_train = df_all["patient_id"].unique()[indices]
    df_all_test = df_all[~df_all["patient_id"].isin(id_train)]
    df_all_train = df_all[df_all["patient_id"].isin(id_train)]
    print(len(df_all_train), len(df_all_test))

    df_mimic = df_all[df_all["attribution"] == "mimic"]
    df_butmimic = df_all[~(df_all["attribution"] == "mimic")]
    df_chexpert = df_all[df_all["attribution"] == "chexpert"]
    df_butchexpert = df_all[~(df_all["attribution"] == "chexpert")]
    df_nih14 = df_all[df_all["attribution"] == "nih14"]
    df_butnih14 = df_all[~(df_all["attribution"] == "nih14")]

    patients_ls = df_mimic["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.95), replace=False
    )
    id_train = df_mimic["patient_id"].unique()[indices]
    df_mimic_test = df_mimic[~df_mimic["patient_id"].isin(id_train)]
    df_mimic_train = df_mimic[df_mimic["patient_id"].isin(id_train)]

    patients_ls = df_butmimic["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.95), replace=False
    )
    id_train = df_butmimic["patient_id"].unique()[indices]
    df_butmimic_test = df_butmimic[~df_butmimic["patient_id"].isin(id_train)]
    df_butmimic_train = df_butmimic[df_butmimic["patient_id"].isin(id_train)]

    patients_ls = df_chexpert["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.95), replace=False
    )
    id_train = df_chexpert["patient_id"].unique()[indices]
    df_chexpert_test = df_chexpert[~df_chexpert["patient_id"].isin(id_train)]
    df_chexpert_train = df_chexpert[df_chexpert["patient_id"].isin(id_train)]

    patients_ls = df_butchexpert["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.96), replace=False
    )
    id_train = df_butchexpert["patient_id"].unique()[indices]
    df_butchexpert_test = df_butchexpert[~df_butchexpert["patient_id"].isin(id_train)]
    df_butchexpert_train = df_butchexpert[df_butchexpert["patient_id"].isin(id_train)]

    patients_ls = df_nih14["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = df_nih14["patient_id"].unique()[indices]
    df_nih14_test = df_nih14[~df_nih14["patient_id"].isin(id_train)]
    df_nih14_train = df_nih14[df_nih14["patient_id"].isin(id_train)]

    patients_ls = df_butnih14["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.975), replace=False
    )
    id_train = df_butnih14["patient_id"].unique()[indices]
    df_butnih14_test = df_butnih14[~df_butnih14["patient_id"].isin(id_train)]
    df_butnih14_train = df_butnih14[df_butnih14["patient_id"].isin(id_train)]

    print(
        len(df_butmimic_train),
        len(df_butmimic_test),
        len(df_mimic_train),
        len(df_mimic_test),
    )
    print(
        len(df_butchexpert_train),
        len(df_butchexpert_test),
        len(df_chexpert_train),
        len(df_chexpert_test),
    )
    print(
        len(df_butnih14_train),
        len(df_butnih14_test),
        len(df_nih14_train),
        len(df_nih14_test),
    )

    for i in np.sort(df_chexpert_test.target.unique()):
        print(i, np.mean(df_chexpert_test.target == i))
    print("ChexpertBut")
    for i in np.sort(df_butchexpert_test.target.unique()):
        print(i, np.mean(df_butchexpert_test.target == i))

    for i in np.sort(df_nih14_test.target.unique()):
        print(i, np.mean(df_nih14_test.target == i))
    print("Nih14But")
    for i in np.sort(df_butnih14_test.target.unique()):
        print(i, np.mean(df_butnih14_test.target == i))

    df_all_train.to_csv(data_dir / "mimic/all_multiclass_train.csv")
    df_all_test.to_csv(data_dir / "mimic/all_multiclass_test.csv")
    df_butmimic_train.to_csv(data_dir / "mimic/butmimic_multiclass_train.csv")
    df_butmimic_test.to_csv(data_dir / "mimic/butmimic_multiclass_test.csv")
    df_mimic_train.to_csv(data_dir / "mimic/mimic_multiclass_train.csv")
    df_mimic_test.to_csv(data_dir / "mimic/mimic_multiclass_test.csv")
    df_butchexpert_train.to_csv(data_dir / "chexpert/butchexpert_multiclass_train.csv")
    df_butchexpert_test.to_csv(data_dir / "chexpert/butchexpert_multiclass_test.csv")
    df_chexpert_train.to_csv(data_dir / "chexpert/chexpert_multiclass_train.csv")
    df_chexpert_test.to_csv(data_dir / "chexpert/chexpert_multiclass_test.csv")
    df_butnih14_train.to_csv(data_dir / "nih14/butnih14_multiclass_train.csv")
    df_butnih14_test.to_csv(data_dir / "nih14/butnih14_multiclass_test.csv")
    df_nih14_train.to_csv(data_dir / "nih14/nih14_multiclass_train.csv")
    df_nih14_test.to_csv(data_dir / "nih14/nih14_multiclass_test.csv")

    print(len(df_all_train), len(df_all_test))

    dflst = []
    for dataset in ["mimic", "nih14", "chexpert"]:
        load = f"{dataset}_multiclass.csv"
        df = pd.read_csv(load)
        df["attribution"] = dataset
        df["filepath"] = str(data_dir / dataset / df["filepath"])
        dflst.append(df)

    xrayall = pd.concat(
        [
            dflst[0][["filepath", "target", "attribution"]],
            dflst[1][["filepath", "target", "attribution"]],
            dflst[2][["filepath", "target", "attribution"]],
        ]
    )

    xrayall.to_csv("xrayall.csv")

    xrayall = pd.concat(
        [
            mimic_df[["filepath", "target", "attribution"]],
            nih_df[["filepath", "target", "attribution"]],
            chex_df[["filepath", "target", "attribution"]],
        ]
    )

    xrayall.to_csv("xrayall.csv")

    for i in range(len(chex_df)):
        _, path = chex_df.iloc[i, chex_df.columns.get_loc("filepath")].split(
            "CheXpert-v1.0-small/"
        )
        chex_df.iloc[i, chex_df.columns.get_loc("filepath")] = path

    chex_df.to_csv("chexpert_multiclass.csv")
