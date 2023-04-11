from __future__ import annotations

import dataclasses
import json
import random
import urllib.request
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import TypeVar

import numpy as np
from deepdiff import DeepDiff
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import get_console
from rich.progress import Progress

import fd_shifts
from fd_shifts import configs
from fd_shifts.experiments import Experiment, get_all_experiments

warnings.filterwarnings("ignore")

GB = 1024 * 1024 * 1024


def get_jobs() -> list[dict[str, str]]:
    with urllib.request.urlopen("http://localhost:3030/jobs") as response:
        records: list[dict[str, str]] = json.loads(response.read())["RECORDS"]
    return records


def is_experiment_running(
    experiment: ValidationResult, jobs: list[dict[str, str]]
) -> bool:
    _experiments = list(
        map(
            lambda j: j["JOB_NAME"],
            filter(lambda j: j["STAT"] in ("RUN", "PEND"), jobs),
        )
    )
    running = (
        str(experiment.experiment.to_path().relative_to("fd-shifts")) in _experiments
    )

    if running:
        print(f"{experiment.experiment.to_path()} is already running")

    return running


NETWORK_DRIVES_PATH = Path("~/NetworkDrives/E130-Personal/Bungert/").expanduser()
BASE_PATH = Path("/media/experiments/")

DTYPES = {
    16: np.float16,
    32: np.float32,
    64: np.float64,
}

test_set_lengths = {
    "cifar10": 806032,
    "cifar100": 806032,
    "super_cifar100": 11400,
    "supercifar": 11400,
    "breeds": 177592,
    "camelyon": 118614,
    "wilds_camelyon": 118614,
    "svhn": 56032,
    "svhn_openset": 56032,
    "animals": 50945,
    "wilds_animals": 50945,
    "wilds_animals_openset": 50945,
    "animals_openset": 50945,
}


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


@dataclass
class ValidationResult:
    experiment: Experiment
    model_exists: bool = False
    config_exists: bool = False
    config_valid: bool = False
    outputs_valid: bool = False
    results_valid: bool = False
    has_job: bool = False
    logs: list[str] = field(default_factory=lambda: [])

    def __str__(self):
        return f"{str(self.experiment.to_path())},{self.model_exists},{self.config_exists},{self.config_valid},{self.outputs_valid}"


def initialize_hydra(overrides: list[str]) -> DictConfig:
    """Takes the place of hydra.main"""
    default_overrides = [
        "hydra.job.num=0",
        "hydra.job.id=0",
        "hydra.runtime.output_dir=.",
        "hydra.hydra_help.hydra_help=''",
    ]

    with initialize_config_module(version_base=None, config_module="fd_shifts.configs"):
        cfg = compose(
            "config", overrides=overrides + default_overrides, return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)

    return cfg


T = TypeVar("T", DictConfig | ListConfig, str | float | bool | int | None)


def apply_to_conf(func: Callable[[str], str], obj: T) -> T:
    match obj:
        case DictConfig():
            for key, val in obj.items_ex(resolve=False):
                obj[key] = apply_to_conf(func, val)
            return obj
        case ListConfig():
            for i, val in enumerate(obj):
                obj[i] = apply_to_conf(func, val)
            return obj
        case str():
            return func(obj)
        case _:
            return obj


def validate_config(experiment: Experiment, config_path: Path):
    dconf = OmegaConf.load(config_path)

    dconf = apply_to_conf(lambda s: s.replace(r"${env:", r"${oc.env:"), dconf)

    dconf.pkgversion = fd_shifts.version()
    dconf.exp.work_dir = str(Path(".").resolve())
    dconf.exp.version = 0

    match experiment.model:
        case "dg":
            model = "deepgamblers"
        case _:
            model = experiment.model

    if not hasattr(dconf.model, "network") or dconf.model.network is None:
        dconf.model["network"] = {"name": "vit"}

    match experiment.dataset:
        case "animals" | "camelyon" | "animals_openset":
            dataset = "wilds_" + experiment.dataset
        case "supercifar":
            dataset = "super_cifar100"
        case _:
            dataset = experiment.dataset

    if model == "vit":
        dataset = dataset + "_384"

    schema = initialize_hydra([f"study={model}", f"data={dataset}_data"])
    schema.model.dropout_rate = dconf.model.dropout_rate
    schema.exp.work_dir = "${hydra:runtime.cwd}"

    match dataset:
        case "svhn" | "wilds_camelyon" | "wilds_animals" | "breeds" | "svhn_openset" | "wilds_animals_openset":
            schema.model.network.backbone = dconf.model.network.backbone

    dconf.eval.query_studies["iid_study"] = schema.eval.query_studies["iid_study"]
    if "openset" in dataset:
        schema.data.kwargs["out_classes"] = random.sample(
            range(schema.data.num_classes), int(0.4 * schema.data.num_classes)
        )
        dconf.data = schema.data
        if model == "vit":
            dconf.exp.name = dconf.exp.name.replace(
                experiment.dataset.replace("_openset", ""), experiment.dataset
            )
        else:
            dconf.exp.group_name = dconf.exp.group_name.replace(
                experiment.dataset.replace("_openset", ""), experiment.dataset
            )

        dconf.eval.query_studies["iid_study"] = schema.eval.query_studies["iid_study"]
        dconf.eval.query_studies["noise_study"] = []
        dconf.eval.query_studies["in_class_study"] = []
        dconf.eval.query_studies["new_class_study"] = []

    oschema: configs.Config = OmegaConf.to_object(schema)
    match dconf.trainer.lr_scheduler.get("name"):
        case "CosineAnnealing":
            dconf.trainer.lr_scheduler._target_ = oschema.trainer.lr_scheduler._target_
            dconf.trainer.lr_scheduler._partial_ = (
                oschema.trainer.lr_scheduler._partial_
            )
            del dconf.trainer.lr_scheduler.name
            if hasattr(dconf.trainer.lr_scheduler, "milestones"):
                del dconf.trainer.lr_scheduler.milestones
            dconf.trainer.lr_scheduler["T_max"] = dconf.trainer.lr_scheduler.max_epochs
            del dconf.trainer.lr_scheduler.max_epochs
            if hasattr(dconf.trainer.lr_scheduler, "gamma"):
                del dconf.trainer.lr_scheduler.gamma
        case "LinearWarmupCosineAnnealing":
            dconf.trainer.lr_scheduler._target_ = oschema.trainer.lr_scheduler._target_
            dconf.trainer.lr_scheduler._partial_ = (
                oschema.trainer.lr_scheduler._partial_
            )
            del dconf.trainer.lr_scheduler.name
        case _:
            pass

    if hasattr(dconf.trainer, "optimizer"):
        match dconf.trainer.optimizer.get("name"):
            case "SGD":
                dconf.trainer.optimizer._target_ = oschema.trainer.optimizer._target_
                dconf.trainer.optimizer._partial_ = oschema.trainer.optimizer._partial_
                del dconf.trainer.optimizer.name
                dconf.trainer.optimizer.lr = dconf.trainer.optimizer.learning_rate
                del dconf.trainer.optimizer.learning_rate
            case _:
                pass

    else:
        dconf.trainer.optimizer = oschema.trainer.optimizer
        dconf.trainer.optimizer.lr = dconf.trainer.learning_rate
        dconf.trainer.optimizer.momentum = dconf.trainer.momentum
        dconf.trainer.optimizer.weight_decay = dconf.trainer.weight_decay
        del dconf.trainer.learning_rate
        del dconf.trainer.momentum
        del dconf.trainer.weight_decay

    dconf.exp.global_seed = 1234

    for key in dconf.eval.query_studies.keys():
        match dconf.eval.query_studies[key]:
            case [*v]:
                dconf.eval.query_studies[key] = list(
                    map(lambda s: s.replace("224", "384"), v)
                )
            case v:
                dconf.eval.query_studies[key] = v.replace("224", "384")

    if hasattr(dconf.trainer, "accelerator"):
        del dconf.trainer.accelerator
    if hasattr(dconf.trainer, "accumulate_grad_batches"):
        del dconf.trainer.accumulate_grad_batches

    if hasattr(dconf.eval.query_studies, "new_class_study"):
        if "tinyimagenet" in dconf.eval.query_studies.new_class_study:
            index = dconf.eval.query_studies.new_class_study.index("tinyimagenet")
            dconf.eval.query_studies.new_class_study.pop(index)

    dconf.test.iid_set_split = "devries"
    dconf.trainer.val_split = configs.ValSplit.devries
    dconf.trainer.batch_size = 128

    dconf.trainer["resume_from_ckpt_confidnet"] = dconf.trainer.get(
        "resume_from_ckpt_confidnet"
    )
    dconf.trainer["num_epochs_backbone"] = dconf.trainer.get("num_epochs_backbone")
    dconf.trainer["learning_rate_confidnet"] = dconf.trainer.get(
        "learning_rate_confidnet"
    )
    dconf.trainer["learning_rate_confidnet_finetune"] = dconf.trainer.get(
        "learning_rate_confidnet_finetune"
    )

    dconf.eval.query_studies["iid_study"] = dconf.eval.query_studies.get(
        "iid_study", "${data.dataset}"
    )
    dconf.eval.query_studies["noise_study"] = dconf.eval.query_studies.get(
        "noise_study", []
    )
    dconf.eval.query_studies["in_class_study"] = dconf.eval.query_studies.get(
        "in_class_study", []
    )
    dconf.eval.query_studies["new_class_study"] = dconf.eval.query_studies.get(
        "new_class_study", []
    )

    dconf.eval.confidence_measures.test = schema.eval.confidence_measures.test

    dconf.test.selection_mode = schema.test.selection_mode
    dconf.eval.test_conf_scaling = schema.eval.test_conf_scaling

    if not hasattr(dconf.model, "fc_dim"):
        dconf.model.fc_dim = schema.model.fc_dim

    if hasattr(dconf.model, "avg_pool"):
        schema.model.avg_pool = dconf.model.avg_pool
    else:
        dconf.model.avg_pool = schema.model.avg_pool

    dconf.model.budget = schema.model.budget
    dconf.model.monitor_mcd_samples = schema.model.monitor_mcd_samples
    dconf.model.test_mcd_samples = schema.model.test_mcd_samples

    if "precision_study" not in str(experiment.group_dir):
        dconf.test.output_precision = 64
        schema.test.output_precision = 64
    elif "16" in str(experiment.group_dir):
        dconf.test.output_precision = 16
        schema.test.output_precision = 16
        dconf.eval.confidence_measures.test = list(
            filter(lambda c: "ext" not in c, dconf.eval.confidence_measures.test)
        )
        schema.eval.confidence_measures.test = list(
            filter(lambda c: "ext" not in c, schema.eval.confidence_measures.test)
        )
        dconf.exp.group_name = f"{dataset.replace('wilds_', '') if experiment.model != 'vit' else 'vit'}_precision_study16"
    elif "32" in str(experiment.group_dir):
        dconf.test.output_precision = 32
        schema.test.output_precision = 32
        dconf.eval.confidence_measures.test = list(
            filter(lambda c: "ext" not in c, dconf.eval.confidence_measures.test)
        )
        schema.eval.confidence_measures.test = list(
            filter(lambda c: "ext" not in c, schema.eval.confidence_measures.test)
        )
        dconf.exp.group_name = f"{dataset.replace('wilds_', '') if experiment.model != 'vit' else 'vit'}_precision_study32"

    dconf.exp.output_paths.test = schema.exp.output_paths.test

    dconf._metadata.object_type = configs.Config

    dconf._metadata.object_type = configs.Config

    def fix_metadata(cfg: DictConfig):
        if hasattr(cfg, "_target_"):
            cfg._metadata.object_type = getattr(configs, cfg._target_.split(".")[-1])
        for k, v in cfg.items():
            match v:
                case DictConfig():
                    fix_metadata(v)
                case _:
                    pass

    fix_metadata(dconf)

    try:
        conf: configs.Config = OmegaConf.to_object(dconf)
        conf.validate()

        def to_dict(obj):
            return json.loads(
                json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)))
            )

        config_diff = DeepDiff(
            to_dict(OmegaConf.to_object(schema)),
            to_dict(conf),
            ignore_order=True,
            exclude_paths={
                "root['hydra']",
                "root['data']['num_workers']",
                "root['data']['data_dir']",
                "root['eval']['confid_metrics']['train']",
                "root['eval']['confid_metrics']['val']",
                "root['eval']['confidence_measures']['val']",
                "root['eval']['confidence_measures']['train']",
                "root['exp']['crossval_ids_path']",
                "root['exp']['dir']",
                "root['exp']['global_seed']",
                "root['exp']['group_dir']",
                "root['exp']['group_name']",
                "root['exp']['log_path']",
                "root['exp']['mode']",
                "root['exp']['name']",
                "root['exp']['output_paths']",
                "root['exp']['version']",
                "root['exp']['version_dir']",
                "root['hydra']['run']['dir']",
                "root['model']['confidnet_fc_dim']",
                "root['model']['dg_reward']",
                "root['model']['fc_dim']",
                "root['model']['network']['imagenet_weights_path']",
                "root['model']['network']['name']",
                "root['model']['network']['save_dg_backbone_path']",
                "root['test']['best_ckpt_path']",
                "root['test']['cf_path']",
                "root['test']['dir']",
                "root['test']['external_confids_output_path']",
                "root['test']['raw_output_path']",
                "root['trainer']['dg_pretrain_epochs']",
                "root['trainer']['do_val']",
                "root['trainer']['lr_scheduler']",
                "root['trainer']['num_epochs']",
                "root['trainer']['num_steps']",
                "root['trainer']['optimizer']",
                "root['trainer']['val_every_n_epoch']",
                "root['trainer']['resume_from_ckpt_confidnet']",
                "root['trainer']['num_epochs_backbone']",
                "root['trainer']['callbacks']['training_stages']",
            },
        )

        if config_diff:
            result = False
        else:
            result = True

        backup_path = config_path.with_suffix(".yaml.bak")
        if backup_path.is_file():
            pass
        else:
            config_path.rename(backup_path)
            OmegaConf.save(dconf, config_path)

        return conf, result

    except Exception as exception:
        raise exception


def validate_outputs(
    experiment: Experiment,
    conf: configs.Config,
    path: Path,
):
    logs = []
    _outputs = [path / "test_results" / "raw_logits.npz"]

    if conf.model.dropout_rate:
        _outputs.append(path / "test_results" / "raw_logits_dist.npz")

    if conf.eval.ext_confid_name and len(
        list(filter(lambda c: "ext" in c, conf.eval.confidence_measures.test))
    ):
        _outputs.append(path / "test_results" / "external_confids.npz")

    if (
        conf.eval.ext_confid_name
        and conf.model.dropout_rate
        and len(list(filter(lambda c: "ext" in c, conf.eval.confidence_measures.test)))
        > 0
    ):
        _outputs.append(path / "test_results" / "external_confids_dist.npz")

    result = True
    for output in _outputs:
        if not output.is_file():
            output = NETWORK_DRIVES_PATH / output.relative_to(BASE_PATH)

        if output.is_file():
            # Don't try and load big files for now, might have to do something with memmap
            if output.stat().st_size > 8 * GB:
                logs.append(f"File {output} is too large to check")
                result = True
                continue

            logs.append(f"{output} exists")

            age_valid = True
            if output.stat().st_mtime < datetime(2022, 8, 1).timestamp():
                logs.append(f"{output} is old")
                age_valid = False

            try:
                content = np.load(output, allow_pickle=True)["arr_0"]
            except Exception as ecpt:
                logs.append(f"{output} unloadable: {ecpt}")
                continue
            dtype_valid = content.dtype == DTYPES[conf.test.output_precision]
            if not dtype_valid:
                logs.append(f"{output} dtype invalid")

            if "external_confids" in str(output):
                dtype_valid = True

            len_valid = content.shape[0] == test_set_lengths[experiment.dataset]

            if "openset" in experiment.dataset:
                len_valid = True

            if not len_valid:
                logs.append(
                    f"{output} len invalid: {content.shape[0]}/{test_set_lengths[experiment.dataset]}"
                )
            result = dtype_valid and len_valid and result and age_valid
        else:
            logs.append(f"{output} does not exist")
            result = False

    return result, logs


def validate_results(
    experiment: Experiment,
    conf: configs.Config,
    path: Path,
):
    _paths: list[Path] = []

    for name, study in conf.eval.query_studies:
        if len(study) == 0:
            continue

        match name:
            case "iid_study":
                _paths.append(path / "test_results" / "analysis_metrics_iid_study.csv")
            case "noise_study":
                _paths.extend(
                    [
                        path / "test_results" / f"analysis_metrics_noise_study_{i}.csv"
                        for i in range(1, 6)
                    ]
                )
            case "new_class_study":
                _paths.extend(
                    map(
                        lambda dset: path
                        / "test_results"
                        / f"analysis_metrics_{name}_{dset}_proposed_mode.csv",
                        study,
                    )
                )
            case _:
                _paths.extend(
                    map(
                        lambda dset: path
                        / "test_results"
                        / f"analysis_metrics_{name}_{dset}.csv",
                        study,
                    )
                )

    logs = []
    result = True
    for p in _paths:
        is_file = p.is_file()

        if not is_file:
            logs.append(f"Result file {p} does not exist")
            result = is_file and result
            continue

        age_valid = True
        if p.stat().st_mtime < datetime(2022, 8, 1).timestamp():
            logs.append(f"{p} is old")
            age_valid = False

        result = is_file and age_valid and result

    return result, logs


def validate_experiment(experiment: Experiment, jobs: list[dict[str, str]]):
    configs.init()
    validation_result = ValidationResult(experiment)
    path = BASE_PATH / experiment.to_path()
    config_path = path / "hydra" / "config.yaml"

    validation_result.has_job = is_experiment_running(validation_result, jobs)

    if config_path.is_file():
        validation_result.config_exists = True

        conf, validation_result.config_valid = validate_config(
            experiment,
            config_path,
        )

        if len(list(path.glob("**/last.ckpt"))) > 0:
            validation_result.model_exists = True

    else:
        return validation_result

    validation_result.outputs_valid, _logs = validate_outputs(
        experiment,
        conf,
        path,
    )
    validation_result.logs.extend(_logs)

    validation_result.results_valid, _logs = validate_results(
        experiment,
        conf,
        path,
    )
    validation_result.logs.extend(_logs)

    validation_result.logs = list(map(lambda s: s + "\n", validation_result.logs))

    return validation_result


if __name__ == "__main__":
    console = get_console()

    validation_results = {}

    # NOTE: `precision_study64` runs are just symlinks to the normal runs,
    # NOTE:     don't need to check them twice
    experiments = list(
        filter(
            lambda e: "precision_study64" not in str(e.to_path()), get_all_experiments()
        )
    )

    experiments = list(
        filter(lambda e: not (e.model != "vit" and e.backbone == "vit"), experiments)
    )

    n_experiments = len(experiments)
    jobs = get_jobs()

    with Progress(console=console) as progress:
        task_id = progress.add_task("[cyan]Working...", total=n_experiments)
        with Pool(
            processes=16,
        ) as pool:
            for valres in pool.imap(
                partial(validate_experiment, jobs=jobs),
                experiments,
                chunksize=8,
            ):
                validation_results[str(valres.experiment.to_path())] = valres
                progress.advance(task_id)

    with open("experiments.json", "wt", encoding="utf-8") as file:
        json.dump(validation_results, file, cls=EnhancedJSONEncoder, indent=2)
