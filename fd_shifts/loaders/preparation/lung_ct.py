import copy
from pathlib import Path
from typing import Callable, Optional

import cv2
import matplotlib.pyplot as plt
import medpy.io as mpy
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def prepare_lidc(data_dir: Path):
    df = pd.read_csv(data_dir / "lidc_idri/id_ood.csv")

    # ## Classification Dataset
    #
    # 1) Binary classification for malignancy
    # 2) Split malignancy ordinal into malignant and benign -> already done
    # 3) Drop all rows where no majority of rates exists
    # 4) Create images from 3D Volumnes
    # 4.1) Read volumne
    # 4.2) Get central slice of nodule in dimension
    # 4.3) Go some fixed distance above and below
    # 4.4) Write slice into image
    # 5) Repeat for all three dimensions
    # 6) splits
    # 6.1) divide patients by iid and ood nodules
    # 6.2) all of the ood nodules go in ood set
    # 6.3) all of these iid nodules go in iid set
    # 6.4) fill up iid testset with only iid patients until 20% of iid remaining trainset
    # 6.5) all remaining patients are the iid trainset
    # 7) Write new filepath and metadata csv for images

    df_dropped = df[df.malignancy_id.notna()]

    # ### Get correct filepath

    raw_fp = df_dropped["Image Save Path"].iloc[10]

    _, fp = raw_fp.split("input/")
    img_nb, _ = fp.split(".nii.gz")
    final_fp = str(data_dir / "lidc_idri/images" / img_nb + ".nii.gz")

    mask_fp_ls = []
    for i in range(4):
        mask_fp = str(data_dir / f"lidc_idri/labels/{img_nb}_0{i}_mask.nii.gz")
        mask_fp_ls.append(mask_fp)

    # Get 3d img

    img_3d, _ = mpy.load(final_fp)

    # Get all 4 masks and sum them up. Then binarize

    mask = np.zeros_like(img_3d)
    for mask_path in mask_fp_ls:
        mask_3d, _ = mpy.load(mask_path)
        mask += mask_3d
    mask = np.where(mask > 0, 1, 0)

    # Compute largest mask in z direction
    # Then get the central slice and the two slices above and below
    # Save them to data storage
    # Currently downscaling from int16 to uint8

    sums_x = np.sum(mask, axis=(1, 2))
    sums_y = np.sum(mask, axis=(0, 2))
    sums_z = np.sum(mask, axis=(0, 1))

    central_slice_idx_x = np.argmax(sums_x)
    central_slice_idx_y = np.argmax(sums_y)
    central_slice_idx_z = np.argmax(sums_z)

    i = -2
    while i <= 2:
        plt.imshow(mask[central_slice_idx_x, :, :])
        print(np.sum(mask[central_slice_idx_x, :, :]))
        print(sums_x[central_slice_idx_x])
        plt.show()

        slice_x = img_3d[central_slice_idx_x + i, :, :]
        plt.imshow(slice_x)
        plt.show()
        io.imsave(data_dir / f"lidc_idri/images2d/{img_nb}_x_{i}.png", slice_x)

        plt.imshow(mask[:, central_slice_idx_y, :])
        print(np.sum(mask[:, central_slice_idx_y, :]))
        print(sums_y[central_slice_idx_y])
        plt.show()

        slice_y = img_3d[:, central_slice_idx_y + i, :]
        plt.imshow(slice_y)
        plt.show()
        io.imsave(data_dir / f"lidc_idri/images2d/{img_nb}_y_{i}.png", slice_y)

        plt.imshow(mask[:, :, central_slice_idx_z])
        print(np.sum(mask[:, :, central_slice_idx_z]))
        print(sums_z[central_slice_idx_z])
        plt.show()

        slice_z = img_3d[:, :, central_slice_idx_z + i]
        plt.imshow(slice_z)
        plt.show()
        io.imsave(data_dir / f"lidc_idri/images2d/{img_nb}_z_{i}.png", slice_z)

        i += 1

    # ## Putting it together

    img_ls = []
    img_nb_ls = []
    img_nb_4_ls = []
    for j in range(len(df_dropped)):
        raw_fp = df_dropped["Image Save Path"].iloc[j]
        _, fp = raw_fp.split("input/")
        img_nb, _ = fp.split(".nii.gz")
        final_fp = str(data_dir / "lidc_idri/images" / img_nb + ".nii.gz")
        mask_fp_ls = []
        for ii in range(4):
            mask_fp = str(data_dir / f"lidc_idri/labels/{img_nb}_0{ii}_mask.nii.gz")
            mask_fp_ls.append(mask_fp)
        img_3d, _ = mpy.load(final_fp)
        mask = np.zeros_like(img_3d)
        for mask_path in mask_fp_ls:
            mask_3d, _ = mpy.load(mask_path)
            mask += mask_3d
        mask = np.where(mask > 0, 1, 0)
        sums_z = np.sum(mask, axis=(0, 1))

        central_slice_idx_z = np.argmax(sums_z)
        i = -2
        img_nb_ls.append(img_nb)
        while i <= 2:
            slice_x = img_3d[central_slice_idx_z + i, :, :]
            save_path = f"/home/l049e/Data/lidc_idri/images2d/{img_nb}_x_{i}.png"
            io.imsave(save_path, slice_x)
            _, filepath = save_path.split("lidc_idri/")
            img_ls.append(filepath)
            img_nb_4_ls.append(img_nb)

            slice_y = img_3d[:, central_slice_idx_z + i, :]

            save_path = f"/home/l049e/Data/lidc_idri/images2d/{img_nb}_y_{i}.png"
            io.imsave(save_path, slice_y)
            _, filepath = save_path.split("lidc_idri/")
            img_ls.append(filepath)
            img_nb_4_ls.append(img_nb)

            slice_z = img_3d[:, :, central_slice_idx_z + i]

            save_path = f"/home/l049e/Data/lidc_idri/images2d/{img_nb}_z_{i}.png"
            io.imsave(save_path, slice_z)
            _, filepath = save_path.split("lidc_idri/")
            img_ls.append(filepath)
            img_nb_4_ls.append(img_nb)

            i += 1

    # By creating a img_nb column in both the filepath list and the original dataframe they can be joined on that columnn

    classification_df = pd.DataFrame()
    classification_df["filepath"] = img_ls
    classification_df["img_nb"] = img_nb_4_ls
    df_dropped["img_nb"] = img_nb_ls
    out_df = pd.merge(df_dropped, classification_df, how="inner")

    for col in out_df.columns:
        if "id" in col:
            print(col)
            print(len(out_df[out_df[col].notna()]))
            print(np.sum(out_df[out_df[col].notna()][col]))

    fp_in = str(data_dir / "lidc_idri" / out_df.filepath.iloc[17])

    out_df["target"] = out_df["malignancy_id"].astype(int)

    out_df.to_csv(data_dir / "lidc_idri/lidc_idri_binaryclass.csv")

    # Getting mean and sd for normalization

    class Lidc_idri(Dataset):
        def __init__(
            self,
            csv: pd.DataFrame,
            train: bool,
            transform: Optional[Callable] = None,
        ):
            self.csv = csv.reset_index(drop=True)
            self.train = train
            self.transform = transform
            self.train_df = self.csv.sample(frac=0.8, random_state=200)
            self.test_df = self.csv.drop(self.train_df.index)
            if self.train:
                self.csv = self.train_df
            elif not self.train:
                self.csv = self.test_df
            self.targets = self.csv.target
            self.imgs = self.csv["filepath"]
            self.samples = self.imgs

        def __len__(self):
            return self.csv.shape[0]

        def __getitem__(self, index):
            row = self.csv.iloc[index]

            image = cv2.imread(row.filepath)
            if self.transform is not None:
                image = Image.fromarray(image)
                image = self.transform(image)
            else:
                image = image.astype(np.float32)
            data = image

            return data, torch.tensor(self.csv.iloc[index].target).long()

    transforms_img = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.2407, 0.2407, 0.2407], std=[0.2437, 0.2437, 0.2437]
            ),
        ]
    )
    df = pd.read_csv(data_dir / "lidc_idri/lidc_idri_binaryclass.csv")
    df["filepath"] = str(data_dir / "lidc_idri" / df["filepath"])

    image_data = Lidc_idri(csv=df, train=True, transform=transforms_img)
    image_data_loader = DataLoader(
        image_data,
        # batch size is whole dataset
        batch_size=len(image_data),
        shuffle=False,
        num_workers=0,
    )

    def mean_std(loader):
        images, lebels = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        return mean, std

    mean, std = mean_std(image_data_loader)
    print("mean and std: \n", mean, std)

    # ## Train and Test split have to be patient exclusive. Roughly 80 20 split
    # Scan ID->Patient  --> Nodules
    # Nodule Index->Nodule --> Slices

    df = pd.read_csv(data_dir / "lidc_idri/lidc_idri_binaryclass.csv")

    df_all = copy.deepcopy(df)

    patients = df_all["Patient ID"].unique()
    np.random.seed(2)
    switchpatients = np.random.choice(patients, 90, replace=False)

    df_all_test = df_all[
        df_all["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]
    df_all_train = df_all[
        ~df_all["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]

    print(len(df_all_train), len(df_all_test))

    df_calnotna = copy.deepcopy(df[~df.calcification_id.isna()])
    ood_patients = df_calnotna[df_calnotna.calcification_id == False][
        "Patient ID"
    ].unique()
    patients_have_ood = df_calnotna[
        df_calnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    patients_have_ood.calcification_id

    ood_test_calcification = patients_have_ood[
        patients_have_ood["calcification_id"] == False
    ]
    iid_testset = patients_have_ood[patients_have_ood["calcification_id"] == True]
    patients_only_iid = df_calnotna[
        ~df_calnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    print(len(patients_only_iid), len(iid_testset), len(ood_test_calcification))
    iid_only_patients = patients_only_iid["Patient ID"].unique()
    np.random.seed(2)
    switchpatients = np.random.choice(iid_only_patients, 48, replace=False)
    len(
        patients_only_iid[
            patients_only_iid["Patient ID"].apply(
                lambda x: x in switchpatients.tolist()
            )
        ]
    )

    iid_testset_2 = patients_only_iid[
        patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]
    iid_train_calcification = patients_only_iid[
        ~patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]

    iid_test_calcification = pd.concat([iid_testset, iid_testset_2])
    print(
        len(iid_train_calcification),
        len(iid_test_calcification),
        len(ood_test_calcification),
    )

    df_spicnotna = copy.deepcopy(df[~df.spiculation_id.isna()])
    ood_patients = df_spicnotna[df_spicnotna.spiculation_id == False][
        "Patient ID"
    ].unique()
    patients_have_ood = df_spicnotna[
        df_spicnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    patients_have_ood.spiculation_id

    ood_test_spiculation = patients_have_ood[
        patients_have_ood["spiculation_id"] == False
    ]
    iid_testset = patients_have_ood[patients_have_ood["spiculation_id"] == True]
    patients_only_iid = df_spicnotna[
        ~df_spicnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    print(len(patients_only_iid), len(iid_testset), len(ood_test_spiculation))
    iid_only_patients = patients_only_iid["Patient ID"].unique()
    np.random.seed(2)
    switchpatients = np.random.choice(iid_only_patients, 49, replace=False)
    len(
        patients_only_iid[
            patients_only_iid["Patient ID"].apply(
                lambda x: x in switchpatients.tolist()
            )
        ]
    )

    iid_testset_2 = patients_only_iid[
        patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]
    iid_train_spiculation = patients_only_iid[
        ~patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]

    iid_test_spiculation = pd.concat([iid_testset, iid_testset_2])
    print(
        len(iid_train_spiculation), len(iid_test_spiculation), len(ood_test_spiculation)
    )

    df_textnotna = copy.deepcopy(df[~df.texture_id.isna()])
    ood_patients = df_textnotna[df_textnotna.texture_id == False]["Patient ID"].unique()
    patients_have_ood = df_textnotna[
        df_textnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    patients_have_ood.texture_id

    ood_test_texture = patients_have_ood[patients_have_ood["texture_id"] == False]
    iid_testset = patients_have_ood[patients_have_ood["texture_id"] == True]
    patients_only_iid = df_textnotna[
        ~df_textnotna["Patient ID"].apply(lambda x: x in ood_patients)
    ]
    print(len(patients_only_iid), len(iid_testset), len(ood_test_texture))
    iid_only_patients = patients_only_iid["Patient ID"].unique()
    np.random.seed(2)
    switchpatients = np.random.choice(iid_only_patients, 49, replace=False)
    len(
        patients_only_iid[
            patients_only_iid["Patient ID"].apply(
                lambda x: x in switchpatients.tolist()
            )
        ]
    )

    iid_testset_2 = patients_only_iid[
        patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]
    iid_train_texture = patients_only_iid[
        ~patients_only_iid["Patient ID"].apply(lambda x: x in switchpatients.tolist())
    ]

    iid_test_texture = pd.concat([iid_testset, iid_testset_2])
    print(len(iid_train_texture), len(iid_test_texture), len(ood_test_texture))

    # ### in the future maybe add other splits like sphericity

    # # Check for Prevailance Shifts

    print(
        np.sum(iid_train_calcification.target) / len(iid_train_calcification.target),
        np.sum(ood_test_calcification.target) / len(ood_test_calcification.target),
    )
    print(
        np.sum(iid_train_spiculation.target) / len(iid_train_spiculation.target),
        np.sum(ood_test_spiculation.target) / len(ood_test_spiculation.target),
    )
    print(
        np.sum(iid_train_texture.target) / len(iid_train_texture.target),
        np.sum(ood_test_texture.target) / len(ood_test_texture.target),
    )

    # # Save Dataset splits

    df_all_train.to_csv(data_dir / "lidc_idri/lidc_idri_binaryclass_all_train.csv")
    df_all_test.to_csv(data_dir / "lidc_idri/lidc_idri_binaryclass_all_test.csv")

    iid_train_calcification.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_calcification_iid_test.csv"
    )
    iid_test_calcification.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_calcification_iid_train.csv"
    )
    ood_test_calcification.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_calcification_ood_test.csv"
    )

    iid_train_spiculation.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_spiculation_iid_train.csv"
    )
    iid_test_spiculation.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_spiculation_iid_test.csv"
    )
    ood_test_spiculation.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_spiculation_ood_test.csv"
    )

    iid_train_texture.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_texture_iid_train.csv"
    )
    iid_test_texture.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_texture_iid_test.csv"
    )
    ood_test_texture.to_csv(
        data_dir / "lidc_idri/lidc_idri_binaryclass_texture_ood_test.csv"
    )
