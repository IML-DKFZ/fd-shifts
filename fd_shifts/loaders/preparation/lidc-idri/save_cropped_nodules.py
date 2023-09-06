import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import numpy as np
import pylidc as pl
import pylidc.utils
from medpy.io import save
from tqdm import tqdm


def main_cli():
    parser = ArgumentParser()
    parser.add_argument(
        "--save_path",
        "-s",
        type=str,
        help="Path to the folder where the cropped nodules will be stored",
        required=True,
    )
    args = parser.parse_args()
    return args


def has_large_mask(nod):
    """
    Checks if the consensus mask is larger than 64 voxels in any dimension. If this is the case, this nodule is
    filtered out
    """
    consensus_mask, _, _ = pylidc.utils.consensus(nod, clevel=0.1)
    max_size_mask = max(consensus_mask.shape)
    if max_size_mask > 64:
        return True


def append_metadata(metadata_nod, nod, first=False):
    features = [
        "subtlety",
        "internal Structure",
        "calcification",
        "sphericity",
        "margin",
        "lobulation",
        "spiculation",
        "texture",
        "malignancy",
    ]
    if first:
        for feature in features:
            metadata_nod[feature] = []
    if nod is not None:
        for feature in features:
            metadata_nod[feature].append(getattr(nod, feature.replace(" ", "")))
    else:
        for feature in features:
            metadata_nod[feature].append(None)


def save_nodules(args: Namespace):
    # Set up the paths to store the data
    save_path = Path(args.save_path)
    images_save_dir = save_path / "images"
    labels_save_dir = save_path / "labels"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    # Get all the Scans
    scans = pl.query(pl.Scan)
    all_metadata = []
    for scan in tqdm(scans):
        nods = scan.cluster_annotations()
        for nod_idx, nod in enumerate(nods):
            # filter nodules that are larger than 64 voxels in any dimension
            if has_large_mask(nod):
                continue
            metadata_nod = {}
            for ann_idx in range(4):
                if ann_idx == 0:
                    # Scan is only saved for first annotation
                    image_size = 64
                    # resample volume and masks to uniform spacing. Returns the interpolation points to resample the
                    # other annotations the same way.
                    vol, mask, irp_pts = nod[ann_idx].uniform_cubic_resample(
                        image_size - 1, return_irp_pts=True
                    )
                    image_save_path = (
                        images_save_dir
                        / f"{str(nod[0].scan.id).zfill(4)}_{str(nod_idx).zfill(2)}.nii.gz"
                    )
                    assert vol.shape == (64, 64, 64)
                    save(vol, str(image_save_path))
                    metadata_nod["Patient ID"] = str(nod[0].scan.patient_id)
                    metadata_nod["Scan ID"] = str(nod[0].scan.id).zfill(4)
                    metadata_nod["Nodule Index"] = str(nod_idx).zfill(2)
                    metadata_nod["Image Save Path"] = image_save_path

                if ann_idx < len(nod):
                    # Resample the other annotations in the same way as the scan
                    mask = nod[ann_idx].uniform_cubic_resample(
                        image_size - 1, resample_vol=False, irp_pts=irp_pts
                    )
                    assert mask.shape == (64, 64, 64)
                    annotation = nod[ann_idx]
                else:
                    # If a nodule has less than four raters, the others are filled with zeros
                    mask = np.zeros([64, 64, 64])
                    annotation = None
                segmentation_save_path = (
                    labels_save_dir
                    / f"{str(nod[0].scan.id).zfill(4)}_{str(nod_idx).zfill(2)}_{str(ann_idx).zfill(2)}_mask.nii.gz"
                )
                save(mask.astype(np.intc), str(segmentation_save_path))
                if ann_idx == 0:
                    metadata_nod["Segmentation Save Paths"] = [segmentation_save_path]
                    append_metadata(metadata_nod, annotation, first=True)
                else:
                    metadata_nod["Segmentation Save Paths"].append(
                        segmentation_save_path
                    )
                    append_metadata(metadata_nod, annotation)
            metadata_series = pd.Series(metadata_nod)
            all_metadata.append(metadata_series)
    metadata = pd.DataFrame(all_metadata)
    metadata.to_csv(save_path / "metadata.csv", index=False)


if __name__ == "__main__":
    cli_args = main_cli()
    save_nodules(cli_args)
