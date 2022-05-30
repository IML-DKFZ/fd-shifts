import logging
import shutil
from functools import cache, reduce
from pathlib import Path
from typing import Any, Iterable, Type

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from loguru import logger
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rich import print
from rich.console import Console
from rich.progress import track

console = Console(force_terminal=True)

logger.remove()  # Remove default 'stderr' handler

logger.add(
    lambda m: console.print(m, end="", highlight=False),
    colorize=True,
    enqueue=True,
    level=logging.DEBUG,
)


@cache
def dataset_list(config: DictConfig):
    datasets: list[Any] = reduce(
        lambda acc, val: acc + [val] if isinstance(val, str) else acc + list(val),
        config.eval.query_studies.values(),
        [],
    )

    # HACK: Clean up old dataset names
    datasets = list(map(lambda dset: dset.replace("224", "384"), datasets))

    if config.eval.val_tuning:
        datasets.insert(0, "val_tuning")

    return np.array(datasets)


def dataset_idx_to_name(indices: pd.Series, config: DictConfig):
    datasets = dataset_list(config)
    return datasets[indices]


def load_raw_outputs(base_path: Path, dtype: Type = np.float64) -> pd.DataFrame:
    # Data format see readme
    output = np.load(base_path)["arr_0"]
    data = pd.DataFrame(
        output,
        columns=[
            *(("softmax", i) for i in range(output.shape[1] - 2)),
            ("label", ""),
            ("dataset", ""),
        ],
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    data = data.astype({("label", ""): int, ("dataset", ""): int})

    return data


base_path = Path("~/Experiments/").expanduser()

paths_64: list[Path] = list(
    filter(
        lambda path: not (path.parent / "external_confids.npz").is_file(),
        (base_path / "vit_64").glob("**/raw_output.npz"),
    )
)

for path in paths_64:
    arr64 = load_raw_outputs(path)
    arr32 = load_raw_outputs(
        base_path / "vit_32" / path.relative_to(base_path / "vit_64"), dtype=np.float16
    )

    path32 = base_path / "vit_32" / path.relative_to(base_path / "vit_64")

    if arr64.shape[0] == arr32.shape[0]:
        continue

    logger.success(
        f"Processing [italic]{path.relative_to(base_path / 'vit_64').parent.parent}"
    )

    config64: DictConfig | ListConfig = OmegaConf.load(
        path.parent.parent / "hydra/config.yaml"
    )
    config32: DictConfig | ListConfig = OmegaConf.load(
        (base_path / "vit_32" / path.relative_to(base_path / "vit_64")).parent.parent
        / "hydra/config.yaml"
    )

    assert isinstance(config64, DictConfig)
    assert isinstance(config32, DictConfig)

    backup_path = path.with_suffix(".npz.bak")

    assert not (path.parent / "raw_output_dist.npz").is_file()

    logger.info(f"Backing up raw_output.npz and config.yaml to {backup_path}")
    if backup_path.is_file():
        logger.warning("Backup exists, using it.")
        arr64 = load_raw_outputs(backup_path)
        config64: DictConfig | ListConfig = OmegaConf.load(path.parent.parent / "hydra/config.yaml.bak")
        assert isinstance(config64, DictConfig)
    else:
        shutil.copy(path, backup_path)
        shutil.copy(path.parent.parent / "hydra/config.yaml", path.parent.parent / "hydra/config.yaml.bak")

    logger.debug(f"Dataset List 64: {dataset_list(config64)}")
    logger.debug(f"Dataset List 32: {dataset_list(config32)}")

    if len(dataset_list(config64)) != len(dataset_list(config32)):
        logger.error("Dataset lists are not equal")

        if (
            (
                base_path / "vit_32" / path.relative_to(base_path / "vit_64")
            ).parent.parent
            / "validation"
        ).is_dir():
            logger.error("Validation available")

    arr64 = arr64.assign(dataset_name=dataset_idx_to_name(arr64.dataset, config64))
    arr32 = arr32.assign(dataset_name=dataset_idx_to_name(arr32.dataset, config32))

    arr64 = arr64[arr64.dataset_name.isin(arr32.dataset_name.unique())].reset_index()

    step = arr64.shape[0] // arr32.shape[0]
    logger.debug(f"Step: {step}")

    best_offset = None

    for offset in range(step):
        logger.debug(f"Trying offset {offset}")

        offset64 = arr64.iloc[offset::step, :].reset_index(drop=True)

        if (arr32.label == offset64.label).all():
            assert (arr32.dataset_name == offset64.dataset_name).all()
            assert (
                arr32.softmax.idxmax(axis=1) != offset64.softmax.idxmax(axis=1)
            ).mean() < 0.01
            logger.error(
                (arr32.softmax.idxmax(axis=1) != offset64.softmax.idxmax(axis=1)).mean()
            )

            best_offset = offset

            dist = np.sqrt(((offset64.softmax - arr32.softmax) ** 2).sum(axis=1))
            if not (dist < np.finfo(np.float16).eps).all():
                logger.warning("Distances are large for valid labels")

            break

    if best_offset is None:
        logger.error("No valid offset found")
    else:
        logger.success(f"[bold green]Step: {step}, Offset: {best_offset}")

    out_arr = (
        arr64.loc[best_offset::step, ["softmax", "label", "dataset"]]
        .reset_index(drop=True)
        .assign(dataset=arr32.dataset)
        .values
    )

    config64.eval.query_studies = config32.eval.query_studies
    config64.eval.confidence_measures = config32.eval.confidence_measures
    assert arr32.loc[:, ["softmax", "label", "dataset"]].shape == out_arr.shape

    np.savez_compressed(path, out_arr)
    shutil.copy(path32.parent / "external_confids.npz",  path.parent)
    OmegaConf.save(config64, path.parent.parent / "hydra/config.yaml")

    if (path32.parent / "raw_output_dist.npz").is_file():
        shutil.copy(path32.parent / "raw_output_dist.npz",  path.parent)
        shutil.copy(path32.parent / "external_confids_dist.npz",  path.parent)
