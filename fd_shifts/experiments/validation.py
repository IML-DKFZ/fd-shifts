import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
from deepdiff import DeepDiff
from hydra import compose, initialize_config_module
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf, resolvers
from rich import inspect
from rich.console import Console
from rich.pretty import pretty_repr
from rich.progress import Progress

import fd_shifts
from fd_shifts import configs
from fd_shifts.experiments import Experiment, get_all_experiments

BASE_PATH = Path("~/Experiments/").expanduser()

test_set_lengths = {
    "cifar10": 806032,
    "cifar100": 806032,
    "super_cifar100": 11400,
    "supercifar": 11400,
    "breeds": 177592,
    "camelyon": 118614,
    "wilds_camelyon": 118614,
    "svhn": 56032,
    "animals": 50945,
    "wilds_animals": 50945,
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
    exists: bool = False
    config_valid: bool = False
    outputs_valid: bool = False
    logs: list[str] = field(default_factory=lambda: [])

    def __str__(self):
        return f"{str(self.experiment.to_path())},{self.exists},{self.config_valid},{self.outputs_valid}"


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


def validate_experiment(experiment: Experiment):
    validation_result = ValidationResult(experiment)
    path = BASE_PATH / experiment.to_path()
    config_path = path / "hydra" / "config.yaml"

    logger = fd_shifts.logger.bind(experiment=str(experiment.to_path()))

    with logger.catch():
        if config_path.is_file():
            logger.info("Config exists")
            validation_result.exists = True

            dconf = OmegaConf.load(config_path)

            dconf = apply_to_conf(lambda s: s.replace(r"${env:", r"${oc.env:"), dconf)

            dconf.pkgversion = fd_shifts.version()
            dconf.exp.work_dir = str(Path(".").resolve())
            dconf.exp.version = 0

            # logger.debug("LR scheduler: {}", dconf.trainer.lr_scheduler.name)

            match dconf.trainer.lr_scheduler.get("name"):
                case "CosineAnnealing":
                    del dconf.trainer.lr_scheduler.name
                    if hasattr(dconf.trainer.lr_scheduler, "milestones"):
                        del dconf.trainer.lr_scheduler.milestones
                    dconf.trainer.lr_scheduler[
                        "T_max"
                    ] = dconf.trainer.lr_scheduler.max_epochs
                    del dconf.trainer.lr_scheduler.max_epochs
                    if hasattr(dconf.trainer.lr_scheduler, "gamma"):
                        del dconf.trainer.lr_scheduler.gamma
                case "LinearWarmupCosineAnnealing":
                    del dconf.trainer.lr_scheduler.name
                case _:
                    # raise NotImplementedError
                    pass

            if hasattr(dconf.trainer, "optimizer"):
                match dconf.trainer.optimizer.get("name"):
                    case "SGD":
                        del dconf.trainer.optimizer.name
                        dconf.trainer.optimizer.lr = (
                            dconf.trainer.optimizer.learning_rate
                        )
                        del dconf.trainer.optimizer.learning_rate
                    case _:
                        # raise NotImplementedError
                        pass

            else:
                dconf.trainer["optimizer"] = DictConfig(
                    {
                        "lr": dconf.trainer.learning_rate,
                        "momentum": dconf.trainer.momentum,
                        "weight_decay": dconf.trainer.weight_decay,
                    }
                )
                del dconf.trainer.learning_rate
                del dconf.trainer.momentum
                del dconf.trainer.weight_decay

            match experiment.model:
                case "dg":
                    model = "deepgamblers"
                case _:
                    model = experiment.model

            if not hasattr(dconf.model, "network") or dconf.model.network is None:
                dconf.model["network"] = {"name": "vit"}

            match experiment.dataset:
                case "animals" | "camelyon":
                    dataset = "wilds_" + experiment.dataset
                case "supercifar":
                    dataset = "super_cifar100"
                case _:
                    dataset = experiment.dataset

            if model == "vit":
                dataset = dataset + "_384"

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
                    index = dconf.eval.query_studies.new_class_study.index(
                        "tinyimagenet"
                    )
                    dconf.eval.query_studies.new_class_study.pop(index)

            dconf.test.iid_set_split = "devries"
            dconf.trainer.val_split = configs.ValSplit.devries
            dconf.trainer.batch_size = 128

            dconf.trainer["resume_from_ckpt_confidnet"] = dconf.trainer.get(
                "resume_from_ckpt_confidnet"
            )
            dconf.trainer["num_epochs_backbone"] = dconf.trainer.get(
                "num_epochs_backbone"
            )
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

            schema = initialize_hydra([f"study={model}", f"data={dataset}_data"])
            schema.model.dropout_rate = dconf.model.dropout_rate
            schema.exp.work_dir = "${hydra:runtime.cwd}"

            dconf.eval.confidence_measures.test = schema.eval.confidence_measures.test

            dconf.test.selection_mode = schema.test.selection_mode
            dconf.eval.test_conf_scaling = schema.eval.test_conf_scaling

            dconf.model.fc_dim = schema.model.fc_dim
            dconf.model.avg_pool = schema.model.avg_pool
            dconf.model.monitor_mcd_samples = schema.model.monitor_mcd_samples
            dconf.model.test_mcd_samples = schema.model.test_mcd_samples

            dconf.exp.output_paths.test = schema.exp.output_paths.test

            dconf._metadata.object_type = configs.Config

            try:
                conf: configs.Config = OmegaConf.to_object(dconf)
                conf.validate()

                def to_dict(obj):
                    return json.loads(
                        json.dumps(
                            obj, default=lambda o: getattr(o, "__dict__", str(o))
                        )
                    )

                config_diff = DeepDiff(
                    to_dict(OmegaConf.to_object(schema)),
                    to_dict(conf),
                    ignore_order=True,
                    exclude_paths={
                        "root['hydra']",
                        "root['data']['num_workers']",
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
                    },
                )

                if config_diff:
                    logger.warning(
                        "Changes to default config: \n{}",
                        config_diff.pretty(),
                    )
                    validation_result.config_valid = False
                else:
                    validation_result.config_valid = True

                backup_path = config_path.with_suffix(".yaml.bak")
                if backup_path.is_file():
                    logger.warning("Backup config exists, not overwriting config")
                else:
                    config_path.rename(backup_path)
                    OmegaConf.save(dconf, config_path)

            except Exception as exception:
                logger.error(
                    DeepDiff(
                        OmegaConf.to_container(schema),
                        OmegaConf.to_container(dconf),
                        ignore_order=True,
                        exclude_paths={
                            "root['data']['num_workers']",
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
                            "root['trainer']['lr_scheduler']['T_max']",
                            "root['trainer']['lr_scheduler']['max_epochs']",
                            "root['trainer']['num_epochs']",
                            "root['trainer']['num_steps']",
                            "root['trainer']['optimizer']",
                            "root['trainer']['val_every_n_epoch']",
                        },
                    ).pretty()
                )
                raise exception

        else:
            logger.error("Config does not exist")
            return validation_result

        _outputs = [path / "test_results" / "raw_logits.npz"]

        if conf.model.dropout_rate:
            _outputs.append(path / "test_results" / "raw_logits_dist.npz")

        if conf.eval.ext_confid_name:
            _outputs.append(path / "test_results" / "external_confids.npz")

        if conf.eval.ext_confid_name and conf.model.dropout_rate:
            _outputs.append(path / "test_results" / "external_confids_dist.npz")

        validation_result.outputs_valid = True
        for output in _outputs:
            if output.is_file():
                logger.info("{} exists", output.stem)
                try:
                    content = np.load(output)["arr_0"]
                except:
                    logger.error("{} unloadable", output.stem)
                    validation_result.outputs_valid = False
                    continue
                dtype_valid = content.dtype == np.float64
                if not dtype_valid:
                    logger.error("{} dtype invalid", output.stem)
                len_valid = content.shape[0] == test_set_lengths[experiment.dataset]
                if not len_valid:
                    logger.error(
                        "{} len invalid: {}/{}",
                        output.stem,
                        content.shape[0],
                        test_set_lengths[experiment.dataset],
                    )
                validation_result.outputs_valid = (
                    dtype_valid and len_valid and validation_result.outputs_valid
                )
            else:
                logger.error("{} does not exist", output.stem)
                validation_result.outputs_valid = False

    return validation_result


if __name__ == "__main__":
    console = Console(stderr=True)
    fd_shifts.logger.remove()  # Remove default 'stderr' handler

    format_log = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[experiment]}</cyan> - <level>{message}</level>"
    )

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    fd_shifts.logger.add(
        lambda m: console.print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="INFO",
        format=format_log,
    )
    fd_shifts.logger.add(
        "{time}_validation.log", enqueue=True, level="INFO", format=format_log
    )

    _logs = {}

    def sink(message):
        exp = message.record["extra"]["experiment"]

        if not exp in _logs:
            _logs[exp] = []

        _logs[exp].append(message)

    fd_shifts.logger.add(sink, enqueue=True, level="INFO")

    configs.init()
    validation_results = {}

    # for experiment in get_all_experiments():
    #     validation_results[str(experiment.to_path())] = validate_experiment(experiment)
    #     break

    with Progress(console=console) as progress:
        task_id = progress.add_task(
            "[cyan]Working...", total=len(list(get_all_experiments()))
        )
        with Pool(
            processes=16,
            # initializer=set_logger,
            # initargs=(logger,),
            # maxtasksperchild=1,
        ) as pool:
            for valres in pool.imap_unordered(
                validate_experiment,
                get_all_experiments(),
            ):
                validation_results[str(valres.experiment.to_path())] = valres
                progress.advance(task_id)

    with open("experiments.csv", "wt", encoding="utf-8") as file:
        file.writelines(
            map(lambda experiment: f"{experiment}\n", validation_results.values())
        )

    for k, v in _logs.items():
        validation_results[k].logs = v

    with open("experiments.json", "wt", encoding="utf-8") as file:
        json.dump(validation_results, file, cls=EnhancedJSONEncoder, indent=2)
