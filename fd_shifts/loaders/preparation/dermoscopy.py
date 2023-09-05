import os
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd


def __prepare_hamm(data_dir: Path):
    ham = pd.read_csv(data_dir / "ham10000/HAM10000_metadata.csv")

    # Classes Melanoma(mel), Basal Cell Carcinoma (bcc) and Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec) are mapped to malignant
    ham["benign_malignant"] = "benign"
    ham["target"] = 0
    for i in range(len(ham)):
        if ham.iloc[i, ham.columns.get_loc("dx")] in ["mel", "bcc", "akiec"]:
            ham.iloc[i, ham.columns.get_loc("benign_malignant")] = "malignant"
            ham.iloc[i, ham.columns.get_loc("target")] = "1"

    # Highest level split: Lesion id
    ham = ham.sort_values("image_id")
    ham["filepath"] = ham["image_id"]
    for i in range(len(ham)):
        part = 1
        if i > 4999:
            part = 2
        imgname = ham["image_id"].iloc[i]
        filepath = f"ham10000_images_part_{part}/{imgname}.jpg"
        ham.iloc[i, ham.columns.get_loc("filepath")] = filepath

    ham.to_csv("ham10000_binaryclass")
    return ham


def __prepare_d7p(data_dir: Path):
    d7p = pd.read_csv(data_dir / "d7p/meta/meta.csv")

    d7p.diagnosis.unique()

    # melanoma and basal cell carcinoma are mapped to malignant

    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(d7p.diagnosis.unique()))}

    diagnosis2idxMel = {d: 0 for idx, d in enumerate(sorted(d7p.diagnosis.unique()))}
    for path in [
        "basal cell carcinoma",
        "melanoma",
        "melanoma (0.76 to 1.5 mm)",
        "melanoma (in situ)",
        "melanoma (less than 0.76 mm)",
        "melanoma (more than 1.5 mm)",
        "melanoma metastasis",
    ]:
        diagnosis2idxMel[path] = 1

    d7p["target"] = d7p["diagnosis"].map(diagnosis2idxMel)

    d7p["filepath"] = "images/" + d7p["derm"]

    print(len(d7p), len(d7p.case_num.unique()))

    # All cases are unique. Free split into train and test
    d7p.to_csv("d7p_binaryclass")
    return d7p


def __prepare_isic(data_dir: Path):
    # ISIC 2020 Dataset

    is2020 = pd.read_csv(
        data_dir / "isic_2020/challenge-2020-training_metadata_2022-06-29(1).csv"
    )

    dia2isx = {"benign": 0, "malignant": 1}
    is2020["target"] = is2020["benign_malignant"].map(dia2isx)

    is2020["target"].astype(int).sum()

    is2020["filepath"] = "train/" + is2020["isic_id"] + ".jpg"

    is2020.to_csv("isic2020_binaryclass")

    return is2020


def __prepare_ph2(data_dir: Path):
    ph2 = pd.read_excel(data_dir / "ph2/PH2_dataset.xlsx", header=12)

    ph2.head()

    ph2["benign_malignant"] = "benign"

    for i in range(len(ph2)):
        if ph2.iloc[i, ph2.columns.get_loc("Melanoma")] == "X":
            ph2.iloc[i, ph2.columns.get_loc("benign_malignant")] = "malignant"

    ph2["target"] = 0
    for i in range(len(ph2)):
        if ph2.iloc[i, ph2.columns.get_loc("Melanoma")] == "X":
            ph2.iloc[i, ph2.columns.get_loc("target")] = 1

    ph2["filepath"] = ph2["Image Name"]
    for i in range(len(ph2)):
        imgname = ph2["Image Name"].iloc[i]
        filepath = (
            f"PH2_Dataset_images/{imgname}/{imgname}_Dermoscopic_Image/{imgname}.bmp"
        )
        ph2.iloc[i, ph2.columns.get_loc("filepath")] = filepath

    ph2.to_csv("ph2_binaryclass")
    return ph2


def prepare_dermoscopy(data_dir: Path):
    # # Creating Shifts datasets
    # 1) Leave Out
    # 1.1) Barcelona
    # 1.2) Queensland
    # 1.3) MSKCC
    # 1.4) Vienna
    # 1.5) ph2
    # 1.6) d7p
    # 2) Subclass shift
    # 2.1) ham10000 subclass shift
    # 3) iid for bottom up
    # 3.1) Barcelona
    # 3.2) MSKCC
    # 3.3) ham1000 multiclass

    ham = __prepare_hamm(data_dir)
    d7p = __prepare_d7p(data_dir)
    ph2 = __prepare_ph2(data_dir)
    is2020 = __prepare_isic(data_dir)

    # Introduce unique identifier column in all Datasets -> "patient_id"
    # when there is no unique identifier generate large unique numbers with ds prefix as id

    ham["patient_id"] = ham["lesion_id"]
    ham["attribution"] = "ham10000"
    ham_small = ham[["filepath", "attribution", "patient_id", "target"]]

    d7p["patient_id"] = d7p["case_num"]
    d7p["attribution"] = "d7p"
    d7p_small = d7p[["filepath", "attribution", "patient_id", "target"]]

    ph2["patient_id"] = ph2["Image Name"]
    ph2["attribution"] = "ph2"
    ph2_small = ph2[["filepath", "attribution", "patient_id", "target"]]

    isic_small = is2020[["filepath", "attribution", "patient_id", "target"]]

    print(is2020.columns)
    print(ham.columns)
    print(ph2.columns)
    print(d7p.columns)

    # ### leave out datasets

    df_all = pd.concat([ham_small, d7p_small, ph2_small, isic_small])

    df_all.attribution.unique()

    pd_barcelona = df_all[
        df_all["attribution"]
        == "Department of Dermatology, Hospital Clínic de Barcelona"
    ]
    pd_butbarcelona = df_all[
        ~(
            df_all["attribution"]
            == "Department of Dermatology, Hospital Clínic de Barcelona"
        )
    ]

    pd_d7p = df_all[df_all["attribution"] == "d7p"]
    pd_butd7p = df_all[~(df_all["attribution"] == "d7p")]

    pd_ph2 = df_all[df_all["attribution"] == "ph2"]
    pd_butph2 = df_all[~(df_all["attribution"] == "ph2")]

    pd_vienna = df_all[
        df_all["attribution"]
        == "ViDIR group, Department of Dermatology, Medical University of Vienna"
    ]
    pd_butvienna = df_all[
        ~(
            df_all["attribution"]
            == "ViDIR group, Department of Dermatology, Medical University of Vienna"
        )
    ]

    pd_mskcc = df_all[df_all["attribution"] == "MSKCC"]
    pd_butmskcc = df_all[~(df_all["attribution"] == "MSKCC")]

    pd_pascal = df_all[df_all["attribution"] == "Pascale Guitera"]
    pd_butpascal = df_all[~(df_all["attribution"] == "Pascale Guitera")]

    pd_queensland = df_all[
        df_all["attribution"]
        == "The University of Queensland Diamantina Institute, The University of Queensland, Dermatology Research Centre"
    ]
    pd_butqueensland = df_all[
        ~(
            df_all["attribution"]
            == "The University of Queensland Diamantina Institute, The University of Queensland, Dermatology Research Centre"
        )
    ]

    # create train testsplit on patient level
    # isic barc,vienna,mskcc at get train testsplit

    patients_ls = df_all["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = df_all["patient_id"].unique()[indices]
    df_all_test = df_all[~df_all["patient_id"].isin(id_train)]
    df_all_train = df_all[df_all["patient_id"].isin(id_train)]

    patients_ls = pd_butbarcelona["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butbarcelona["patient_id"].unique()[indices]
    pd_butbarcelona_test = pd_butbarcelona[
        ~pd_butbarcelona["patient_id"].isin(id_train)
    ]
    pd_butbarcelona_train = pd_butbarcelona[
        pd_butbarcelona["patient_id"].isin(id_train)
    ]

    patients_ls = pd_barcelona["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_barcelona["patient_id"].unique()[indices]
    pd_barcelona_test = pd_barcelona[~pd_barcelona["patient_id"].isin(id_train)]
    pd_barcelona_train = pd_barcelona[pd_barcelona["patient_id"].isin(id_train)]

    patients_ls = pd_butd7p["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butd7p["patient_id"].unique()[indices]
    pd_butd7p_test = pd_butd7p[~pd_butd7p["patient_id"].isin(id_train)]
    pd_butd7p_train = pd_butd7p[pd_butd7p["patient_id"].isin(id_train)]

    # small ds only ood
    pd_d7p_test = pd_d7p

    patients_ls = pd_butph2["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butph2["patient_id"].unique()[indices]
    pd_butph2_test = pd_butph2[~pd_butph2["patient_id"].isin(id_train)]
    pd_butph2_train = pd_butph2[pd_butph2["patient_id"].isin(id_train)]

    # small ds only oof

    pd_ph2_test = pd_ph2

    patients_ls = pd_butvienna["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butvienna["patient_id"].unique()[indices]
    pd_butvienna_test = pd_butvienna[~pd_butvienna["patient_id"].isin(id_train)]
    pd_butvienna_train = pd_butvienna[pd_butvienna["patient_id"].isin(id_train)]

    patients_ls = pd_vienna["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_vienna["patient_id"].unique()[indices]
    pd_vienna_test = pd_vienna[~pd_vienna["patient_id"].isin(id_train)]
    pd_vienna_train = pd_vienna[pd_vienna["patient_id"].isin(id_train)]

    patients_ls = pd_butqueensland["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butqueensland["patient_id"].unique()[indices]
    pd_butqueensland_test = pd_butqueensland[
        ~pd_butqueensland["patient_id"].isin(id_train)
    ]
    pd_butqueensland_train = pd_butqueensland[
        pd_butqueensland["patient_id"].isin(id_train)
    ]

    patients_ls = pd_queensland["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_queensland["patient_id"].unique()[indices]
    pd_queensland_test = pd_queensland[~pd_queensland["patient_id"].isin(id_train)]
    pd_queensland_train = pd_queensland[pd_queensland["patient_id"].isin(id_train)]

    patients_ls = pd_butpascal["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butpascal["patient_id"].unique()[indices]
    pd_butpascal_test = pd_butpascal[~pd_butpascal["patient_id"].isin(id_train)]
    pd_butpascal_train = pd_butpascal[pd_butpascal["patient_id"].isin(id_train)]

    # small ds only ood
    pd_pascal_test = pd_pascal

    patients_ls = pd_butmskcc["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_butmskcc["patient_id"].unique()[indices]
    pd_butmskcc_test = pd_butmskcc[~pd_butmskcc["patient_id"].isin(id_train)]
    pd_butmskcc_train = pd_butmskcc[pd_butmskcc["patient_id"].isin(id_train)]

    patients_ls = pd_mskcc["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = pd_mskcc["patient_id"].unique()[indices]
    pd_mskcc_test = pd_mskcc[~pd_mskcc["patient_id"].isin(id_train)]
    pd_mskcc_train = pd_mskcc[pd_mskcc["patient_id"].isin(id_train)]

    for attribution in is2020["attribution"].unique().tolist():
        sum = np.sum(is2020["attribution"] == attribution)
        mal = np.sum(
            is2020[is2020["attribution"] == attribution]["benign_malignant"]
            == "malignant"
        )
        ben = np.sum(
            is2020[is2020["attribution"] == attribution]["benign_malignant"] == "benign"
        )
        print(f"{attribution}: {sum}")
        print(f"mal: {mal}")
        print(f"ben: {ben}")

    print(len(df_all_train), len(df_all_test))
    print(
        len(pd_butmskcc_train),
        len(pd_butmskcc_test),
        len(pd_mskcc_train),
        len(pd_mskcc_test),
    )
    print(len(pd_butpascal_train), len(pd_butpascal_test), len(pd_pascal_test))
    print(
        len(pd_butbarcelona_train),
        len(pd_butbarcelona_test),
        len(pd_barcelona_train),
        len(pd_barcelona_test),
    )
    print(len(pd_butph2_train), len(pd_butph2_test), len(pd_ph2_test))
    print(len(pd_butd7p_train), len(pd_butd7p_test), len(pd_d7p_test))
    print(
        len(pd_butqueensland_train),
        len(pd_butqueensland_test),
        len(pd_queensland_train),
        len(pd_queensland_test),
    )
    print(
        len(pd_butvienna_train),
        len(pd_butvienna_test),
        len(pd_vienna_train),
        len(pd_vienna_test),
    )

    df_all_test.to_csv("~/Data/isic_2020/all_binaryclass_test.csv")
    df_all_train.to_csv("~/Data/isic_2020/all_binaryclass_train.csv")
    pd_butmskcc_train.to_csv("~/Data/isic_2020/butmskcc_binaryclass_train.csv")
    pd_butpascal_train.to_csv("~/Data/isic_2020/butpascal_binaryclass_train.csv")
    pd_butbarcelona_train.to_csv("~/Data/isic_2020/butbarcelona_binaryclass_train.csv")
    pd_butph2_train.to_csv("~/Data/isic_2020/butph2_binaryclass_train.csv")
    pd_butd7p_train.to_csv("~/Data/isic_2020/butd7p_binaryclass_train.csv")
    pd_butqueensland_train.to_csv(
        "~/Data/isic_2020/butqueensland_binaryclass_train.csv"
    )
    pd_butvienna_train.to_csv("~/Data/isic_2020/butvienna_binaryclass_train.csv")
    pd_butmskcc_test.to_csv("~/Data/isic_2020/butmskcc_binaryclass_test.csv")
    pd_butpascal_test.to_csv("~/Data/isic_2020/butpascal_binaryclass_test.csv")
    pd_butbarcelona_test.to_csv("~/Data/isic_2020/butbarcelona_binaryclass_test.csv")
    pd_butph2_test.to_csv("~/Data/isic_2020/butph2_binaryclass_test.csv")
    pd_butd7p_test.to_csv("~/Data/isic_2020/butd7p_binaryclass_test.csv")
    pd_butqueensland_test.to_csv("~/Data/isic_2020/butqueensland_binaryclass_test.csv")
    pd_butvienna_test.to_csv("~/Data/isic_2020/butvienna_binaryclass_test.csv")
    pd_mskcc_train.to_csv("~/Data/isic_2020/mskcc_binaryclass_train.csv")
    pd_barcelona_train.to_csv("~/Data/isic_2020/barcelona_binaryclass_train.csv")
    pd_queensland_train.to_csv("~/Data/isic_2020/queensland_binaryclass_train.csv")
    pd_vienna_train.to_csv("~/Data/isic_2020/vienna_binaryclass_train.csv")
    pd_mskcc_test.to_csv("~/Data/isic_2020/mskcc_binaryclass_test.csv")
    pd_pascal_test.to_csv("~/Data/isic_2020/pascal_binaryclass_test.csv")
    pd_barcelona_test.to_csv("~/Data/isic_2020/barcelona_binaryclass_test.csv")
    pd_ph2_test.to_csv("~/Data/ph2/ph2_binaryclass_test.csv")
    pd_d7p_test.to_csv("~/Data/d7p/d7p_binaryclass_test.csv")
    pd_queensland_test.to_csv("~/Data/isic_2020/queensland_binaryclass_test.csv")
    pd_vienna_test.to_csv("~/Data/isic_2020/vienna_binaryclass_test.csv")

    # ## Ham10000 Multiclass and subclass shifts

    ham_small = ham[ham.dx.isin(["bkl", "akiec"])]
    ham_big = ham[~ham.dx.isin(["bkl,akiec"])]
    print(len(ham_small), len(ham_big))
    ham_small["target"] = ham_small.target.astype(int)

    ham["target"] = ham.target.astype(int)
    ham_multi = copy(ham)
    dia2target = {x: y for y, x in enumerate(ham_multi.dx.unique())}
    ham_multi["target"] = ham_multi.dx.map(dia2target)

    patients_ls = ham_multi["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = ham_multi["patient_id"].unique()[indices]
    ham_multi_test = ham_multi[~ham_multi["patient_id"].isin(id_train)]
    ham_multi_train = ham_multi[ham_multi["patient_id"].isin(id_train)]

    patients_ls = ham_big["patient_id"].unique()
    np.random.seed(2)
    indices = np.random.choice(
        len(patients_ls), int(len(patients_ls) * 0.8), replace=False
    )
    id_train = ham_big["patient_id"].unique()[indices]
    ham_big_test = ham_big[~ham_big["patient_id"].isin(id_train)]
    ham_big_train = ham_big[ham_big["patient_id"].isin(id_train)]

    ham_small_test = ham_small
    print(len(ham_multi_train), len(ham_multi_test))
    print(len(ham_big_train), len(ham_big_test), len(ham_small_test))
    print(
        np.sum(ham_big_train.target.astype(int)) / len(ham_big_train),
        np.sum(ham_small_test.target.astype(int)) / len(ham_small_test),
    )

    ham_multi_train.to_csv("~/Data/ham10000/ham10000_multiclass_train.csv")
    ham_multi_test.to_csv("~/Data/ham10000/ham10000_multiclass_test.csv")
    ham_big_train.to_csv("~/Data/ham10000/ham10000_subbig_train.csv")
    ham_big_test.to_csv("~/Data/ham10000/ham10000_subbig_test.csv")
    ham_small_test.to_csv("~/Data/ham10000/ham10000_subsmall_test.csv")

    root = os.getcwd()
    datasets = ["d7p", "ham10000", "ph2", "isic_2020"]
    dataframes = []
    for dataset in datasets:
        csv_file = f"{root}/{dataset}_binaryclass"
        df_train = pd.read_csv(csv_file)
        df_train["filepath"] = str(data_dir / dataset / df_train["filepath"])
        col_to_keep = ["filepath", "target"]
        df_train = df_train[col_to_keep]
        dataframes.append(df_train)

    df_train = pd.concat(dataframes)
    df_train.to_csv("df_all_dermoscopy.csv")
