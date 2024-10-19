import concurrent.futures
import functools
from itertools import product
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fd_shifts import logger
from fd_shifts.configs import Config
from fd_shifts.experiments.configs import get_experiment_config, list_experiment_configs
from fd_shifts.experiments.tracker import list_bootstrap_analysis_output_files
from fd_shifts.reporting import (
    DATASETS,
    _filter_unused,
    _load_file,
    assign_hparams_from_names,
    filter_best_hparams,
    filter_best_lr,
    rename_confids,
    rename_studies,
    tables,
)
from fd_shifts.reporting.plots_bootstrap import (
    bs_blob_plot,
    bs_box_scatter_plot,
    bs_kendall_tau_comparing_metrics,
    bs_kendall_tau_violin,
    bs_podium_plot,
    bs_significance_map,
    bs_significance_map_colored,
    bs_significance_map_colored_corrected,
)


def _load_bootstrap_experiment(
    name: str,
    filter_study_name: list = None,
    filter_dataset: list = None,
    original_new_class_mode: bool = False,
) -> pd.DataFrame | None:
    from fd_shifts.main import omegaconf_resolve

    config = get_experiment_config(name)
    config = omegaconf_resolve(config)

    if filter_dataset is not None and config.data.dataset not in filter_dataset:
        # handle super-cifar100 experiments
        if (
            filter_dataset == ["cifar100"]
            and config.data.dataset == "super_cifar100"
            and filter_study_name == ["in_class_study"]
        ):
            filter_study_name = ["iid_study"]
        else:
            return

    data = list(
        map(
            functools.partial(_load_file, config, name),
            list_bootstrap_analysis_output_files(
                config, filter_study_name, original_new_class_mode
            ),
        )
    )

    if len(data) == 0 or any(map(lambda d: d is None, data)):
        return

    data = pd.concat(data)  # type: ignore
    data = (
        data.assign(
            experiment=config.data.dataset + ("vit" if "vit" in name else ""),
            run=int(name.split("run")[1].split("_")[0]),
            dropout=config.model.dropout_rate,
            rew=config.model.dg_reward if config.model.dg_reward is not None else 0,
            lr=config.trainer.optimizer.init_args["init_args"]["lr"],
        )
        .dropna(subset=["name", "model"])
        .drop_duplicates(
            subset=["name", "study", "model", "network", "confid", "bootstrap_index"]
        )
    )

    return data


def load_all(
    filter_study_name: list = None,
    filter_dataset: list = None,
    original_new_class_mode: bool = False,
    include_vit: bool = True,
):
    dataframes = []
    # TODO: make this async
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        dataframes = list(
            filter(
                lambda d: d is not None,
                executor.map(
                    functools.partial(
                        _load_bootstrap_experiment,
                        filter_study_name=filter_study_name,
                        filter_dataset=filter_dataset,
                        original_new_class_mode=original_new_class_mode,
                    ),
                    filter(
                        (
                            (lambda exp: True)
                            if include_vit
                            else lambda exp: not exp.startswith("vit")
                        ),
                        list_experiment_configs(),
                    ),
                ),
            )
        )

    data = pd.concat(dataframes)  # type: ignore
    data = data.loc[~data["study"].str.contains("tinyimagenet_original")]
    data = data.loc[~data["study"].str.contains("tinyimagenet_proposed")]

    # data = data.query(
    #     'not (experiment in ["cifar10", "cifar100", "super_cifar100"]'
    #     'and not name.str.contains("vgg13"))'
    # )

    data = data.query(
        'not ((experiment.str.contains("super_cifar100")'
        ")"
        'and not (study == "iid_study"))'
    )

    data = data.query(
        'not (experiment.str.contains("openset")' 'and study.str.contains("iid_study"))'
    )

    data = data.assign(study=data.experiment + "_" + data.study)

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100",
            "cifar100_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100", "cifar100"
        ),
    )

    data = data.assign(
        study=data.study.mask(
            data.experiment == "super_cifar100vit",
            "cifar100vit_in_class_study_superclasses",
        ),
        experiment=data.experiment.mask(
            data.experiment == "super_cifar100vit", "cifar100vit"
        ),
    )

    data = data.assign(ece=data.ece.mask(data.ece < 0))

    return data


def create_plots_per_study(
    study: str,
    dset: str,
    metrics: list,
    out_dir: Path,
    original_new_class_mode: bool = False,
    metric_hparam_search: str | None = None,
    new_class_study_dset: str | None = None,
):
    logger.info(f"Reporting bootstrap results for dataset '{dset}', study '{study}'")

    data_raw = load_all(
        filter_study_name=[study],
        filter_dataset=[dset],
        original_new_class_mode=original_new_class_mode,
        include_vit=False,
    )

    study_name = study  # used for file names

    if study == "new_class_study":
        if new_class_study_dset is None:
            for nc_study in data_raw["study"].unique():
                logger.info(f"Creating plots for {nc_study}...")
                create_plots_per_study(
                    study,
                    dset,
                    metrics,
                    out_dir,
                    original_new_class_mode,
                    metric_hparam_search,
                    str(nc_study),
                )
            return
        else:
            study_name = new_class_study_dset  # used for file names
            data_raw = data_raw[
                data_raw["study"].str.contains(new_class_study_dset)
                | data_raw["study"].str.contains("val_tuning")
            ]

    if not any(data_raw["study"].str.contains(study)):
        logger.info(
            f"No {study} data found for dataset {dset}, new_class_study_dset {new_class_study_dset}. Skipping."
        )
        return

    data_raw = assign_hparams_from_names(data_raw)

    for metric in metrics:
        metric_to_optimize = (
            metric if metric_hparam_search is None else metric_hparam_search
        )
        data, selection_df = filter_best_lr(data_raw, metric=metric_to_optimize)
        selection_df.to_csv(
            out_dir / f"filter_best_lr_{dset}_{metric_to_optimize}.csv", decimal="."
        )
        data, selection_df = filter_best_hparams(
            data, bootstrap_analysis=True, metric=metric_to_optimize
        )
        selection_df.to_csv(
            out_dir / f"filter_best_hparams_{dset}_{metric_to_optimize}.csv",
            decimal=".",
        )
        data = _filter_unused(data)

        # Filter MCD data
        # data = data[~data.confid.str.contains("mcd")]

        data = rename_confids(data)
        data = rename_studies(data)

        data = data[data.confid.isin(CONFIDS_TO_REPORT)]

        logger.info("Removing 'val_tuning' studies and aggregating noise studies")
        data = data[~data["study"].str.contains("val_tuning")]

        # Rename the <dset>_noise_study_X entries to the same value such that the groupby
        # operation automatically averages over the 5 different noise levels.
        if study == "noise_study":
            data["study"] = "noise_study"

        # First, do all plots without aggregation across runs, then aggregate
        # for aggregate_runs in (False, True):
        for aggregate_runs in (True,):
            if aggregate_runs:
                data, _ = tables.aggregate_over_runs(data, metric_columns=metrics)
                group_columns = ["bootstrap_index"]
                blob_dir = out_dir / "blob_run_avg"
                podium_dir = out_dir / "podium_run_avg"
                box_dir = out_dir / "box_run_avg"
                significance_map_dir = out_dir / "significance_map_run_avg"
                kendall_violin_dir = out_dir / "kendall_violin_run_avg"
            else:
                data = data[["confid", "study", "run", "bootstrap_index"] + metrics]
                group_columns = ["bootstrap_index", "run"]
                blob_dir = out_dir / "blob"
                podium_dir = out_dir / "podium"
                box_dir = out_dir / "box"
                significance_map_dir = out_dir / "significance_map"
                kendall_violin_dir = out_dir / "kendall_violin"

            blob_dir.mkdir(exist_ok=True)
            podium_dir.mkdir(exist_ok=True)
            box_dir.mkdir(exist_ok=True)
            significance_map_dir.mkdir(exist_ok=True)
            kendall_violin_dir.mkdir(exist_ok=True)

            # Compute method ranking per bootstrap sample (and run if aggregate_runs=False)
            data["rank"] = data.groupby(group_columns)[metric].rank(method="min")

            # Compute ranking histogram per method
            histograms = (
                data.groupby("confid")["rank"].value_counts().unstack(fill_value=0)
            )

            # Sort methods by mean rank
            histograms["mean_rank"] = (histograms.columns * histograms).sum(
                axis=1
            ) / histograms.sum(axis=1)
            histograms["median_rank"] = (
                data.groupby("confid")["rank"].median().astype(int)
            )
            histograms = histograms.sort_values(by=["mean_rank", "median_rank"])

            medians = histograms.median_rank
            histograms = histograms.drop(columns="mean_rank")
            histograms = histograms.drop(columns="median_rank")

            filename = f"blob_plot_{dset}_{study_name}_{metric}.pdf"
            bs_blob_plot(
                histograms=histograms,
                medians=medians,
                out_dir=blob_dir,
                filename=filename,
            )

            filename = f"podium_plot_{dset}_{study_name}_{metric}.pdf"
            bs_podium_plot(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=podium_dir,
                filename=filename,
            )

            filename = f"box_plot_{dset}_{study_name}_{metric}.pdf"
            bs_box_scatter_plot(
                data=data,
                metric=metric,
                out_dir=box_dir,
                filename=filename,
            )

            filename = f"significance_map_{dset}_{study_name}_{metric}.pdf"
            bs_significance_map(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
            )

            filename = f"colored_significance_map_{dset}_{study_name}_{metric}.pdf"
            bs_significance_map_colored(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
                no_labels=True,
                flip_horizontally=(metric == "aurc"),
            )

            filename = f"colored_significance_map_holm_{dset}_{study_name}_{metric}.pdf"
            bs_significance_map_colored_corrected(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
                no_labels=True,
                flip_horizontally=(metric == "aurc"),
                correction="holm",
            )

            filename = (
                f"colored_significance_map_bonferroni_{dset}_{study_name}_{metric}.pdf"
            )
            bs_significance_map_colored_corrected(
                data=data,
                metric=metric,
                histograms=histograms,
                out_dir=significance_map_dir,
                filename=filename,
                no_labels=True,
                flip_horizontally=(metric == "aurc"),
                correction="bonferroni",
            )

            if "aurc" in metrics and "augrc" in metrics:
                filename = f"kendall_violin_{dset}_{study_name}_{metric}.pdf"
                bs_kendall_tau_violin(
                    data=data,
                    metric=metric,
                    histograms=histograms,
                    out_dir=kendall_violin_dir,
                    filename=filename,
                )


def create_kendall_tau_plot(out_dir: Path):
    logger.info(f"Performing iid-study kendall tau analysis across datasets...")

    data_raw = load_all(filter_study_name=["iid_study"], include_vit=False)
    data_raw = assign_hparams_from_names(data_raw)

    processed_data = {}
    processed_histograms = {}

    # First, do all plots without aggregation across runs, then aggregate
    for aggregate_runs in (False, True):
        for metric in ["aurc", "augrc"]:
            data, _ = filter_best_lr(data_raw, metric=metric)
            data, _ = filter_best_hparams(data, bootstrap_analysis=True, metric=metric)
            data = _filter_unused(data)
            data = rename_confids(data)
            data = rename_studies(data)

            data = data[data.confid.isin(CONFIDS_TO_REPORT)]

            logger.info("Removing 'val_tuning' studies")
            data = data[~data["study"].str.contains("val_tuning")]

            if aggregate_runs:
                data, _ = tables.aggregate_over_runs(
                    data, metric_columns=["aurc", "augrc"]
                )
                group_columns = ["bootstrap_index"]
            else:
                data = data[
                    ["confid", "study", "run", "bootstrap_index"] + ["aurc", "augrc"]
                ]
                group_columns = ["bootstrap_index", "run"]

            # Compute method ranking per bootstrap sample (and run if aggregate_runs=False)
            data["rank"] = data.groupby(group_columns)[metric].rank(method="min")
            # Compute ranking histogram per method
            histograms = (
                data.groupby("confid")["rank"].value_counts().unstack(fill_value=0)
            )
            # Sort methods by mean rank
            histograms["mean_rank"] = (histograms.columns * histograms).sum(
                axis=1
            ) / histograms.sum(axis=1)
            histograms["median_rank"] = (
                data.groupby("confid")["rank"].median().astype(int)
            )
            # histograms = histograms.sort_values(by=["median_rank", "mean_rank"])
            histograms = histograms.sort_values(by=["mean_rank", "median_rank"])

            processed_data[metric] = data
            processed_histograms[metric] = histograms

        if aggregate_runs:
            filename = "kendall_tau_iid_aurc_vs_augrc_run_avg.pdf"
        else:
            filename = "kendall_tau_iid_aurc_vs_augrc.pdf"
        bs_kendall_tau_comparing_metrics(
            processed_data,
            processed_histograms,
            out_dir,
            filename,
        )


def ranking_change_arrows(out_dir: Path):
    """"""
    import matplotlib.pyplot as plt

    _DATASETS = ["wilds_animals", "wilds_camelyon", "cifar10", "breeds"]

    data_raw = load_all(filter_dataset=_DATASETS, include_vit=False)
    data_raw = assign_hparams_from_names(data_raw)

    mean_rank_dict = {}
    median_rank_dict = {}

    for metric in ["aurc", "augrc"]:
        data, _ = filter_best_lr(data_raw, metric=metric)
        data, _ = filter_best_hparams(data, bootstrap_analysis=True, metric=metric)
        data = _filter_unused(data)
        data = rename_confids(data)
        data = rename_studies(data)
        data = data[data.confid.isin(CONFIDS_TO_REPORT)]
        logger.info("Removing 'val_tuning' studies")
        data = data[~data["study"].str.contains("val_tuning")]

        # Aggregate metric values over runs (should we instead rank then aggregate?)
        data, _ = tables.aggregate_over_runs(data, metric_columns=["aurc", "augrc"])

        # Aggregate noise studies
        if "cifar10" in _DATASETS or "cifar100" in _DATASETS:
            data["study"] = data["study"].replace(
                [
                    s
                    for s in list(data["study"].unique())
                    if (s.startswith("cifar10") and "noise_study" in s)
                ],
                "cifar10_noise_study",
            )
            data = (
                data.groupby(["study", "confid", "bootstrap_index"])
                .mean()
                .reset_index()
            )

        data["rank"] = data.groupby(["bootstrap_index", "study"])[metric].rank(
            method="min"
        )
        mean_ranks = data.groupby(["study", "confid"])["rank"].mean()
        median_ranks = data.groupby(["study", "confid"])["rank"].median()

        studies = list(data["study"].unique())

        mean_rank_dict[metric] = mean_ranks.reset_index(level=["study"])
        median_rank_dict[metric] = median_ranks.reset_index(level=["study"])
        del data

    del data_raw

    confid_to_label = {c: f"C{i+1}" for i, c in enumerate(sorted(CONFIDS_TO_REPORT))}
    n_confid = len(confid_to_label)

    for c, l in confid_to_label.items():
        print(f"{l}: {c}")

    arrow_offset = 0.01
    column_distance = 0.1

    # DataFrame with columns: study, rank ; index: confid
    mean_rank_aurc = mean_rank_dict["aurc"]
    mean_rank_augrc = mean_rank_dict["augrc"]

    out_dir = out_dir / "ranking-changes"
    out_dir.mkdir(exist_ok=True)

    print(f"{mean_rank_aurc = }")

    for s in studies:
        try:
            # Rank CSFs by average rank
            ranks_aurc = mean_rank_aurc[mean_rank_aurc["study"] == s]["rank"].rank()
            ranks_augrc = mean_rank_augrc[mean_rank_augrc["study"] == s]["rank"].rank()
            # -> Series objects with confid index and rank values

            for confid in confid_to_label:
                y1 = ranks_aurc[confid]
                y2 = ranks_augrc[confid]

                if y1 != y2:
                    plt.arrow(
                        x=arrow_offset,
                        y=n_confid - y1,
                        dx=column_distance - 2 * arrow_offset,
                        dy=y1 - y2,
                        length_includes_head=True,
                        width=0.00015,
                        head_width=0.01,
                        head_length=0.05,
                        overhang=0.1,
                        color="tab:red",
                    )
                plt.text(
                    x=0,
                    y=n_confid - y1,
                    s=confid_to_label[confid],
                    horizontalalignment="right",
                    verticalalignment="center",
                )
                plt.text(
                    x=column_distance,
                    y=n_confid - y2,
                    s=confid_to_label[confid],
                    horizontalalignment="left",
                    verticalalignment="center",
                )

            plt.xlim(-0.5, 0.6)
            plt.ylim(-0.2, n_confid - 0.4)
            plt.axis("off")
            plt.savefig(
                out_dir / f"bootstrap_ranking_change_arrows_{s}.pdf",
                bbox_inches="tight",
            )
            plt.close()

        except Exception as err:
            logger.info(f"ERROR for study {s}: '{str(err)}'")
            continue


CONFIDS_TO_REPORT = [
    "MSR",
    "MLS",
    "PE",
    "MCD-MSR",
    "MCD-PE",
    "MCD-EE",
    "DG-MCD-MSR",
    "ConfidNet",
    "DG-Res",
    "Devries et al.",
    "TEMP-MLS",
    "DG-PE",
    "DG-TEMP-MLS",
]


def report_bootstrap_results(
    out_path: str | Path = "./output/bootstrap", metric_hparam_search: str = None
):
    """"""
    if metric_hparam_search is not None:
        out_path = str(out_path) + f"-optimized-{metric_hparam_search}"

    data_dir: Path = Path(out_path).expanduser().resolve()
    data_dir.mkdir(exist_ok=True, parents=True)

    # Select all studies, datasets, and metrics
    datasets = [d for d in DATASETS if d != "super_cifar100"]
    studies = ["iid_study", "in_class_study", "new_class_study", "noise_study"]
    metrics = ["aurc", "augrc"]

    logger.info(
        f"Reporting bootstrap results for datasets '{datasets}', studies '{studies}'.\n"
        f"output directory: {data_dir}"
    )

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            # Submit tasks to the executor
            future_to_arg = {
                executor.submit(
                    create_plots_per_study,
                    study=study,
                    dset=dset,
                    metrics=metrics,
                    out_dir=data_dir,
                    original_new_class_mode=False,
                    metric_hparam_search=metric_hparam_search,
                ): dict(study=study, dset=dset)
                for dset, study in product(datasets, studies)
            }

            try:
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_arg),
                    total=len(future_to_arg),
                ):
                    arg = future_to_arg[future]
                    # Get the result from the future (this will raise an exception if the
                    # function call raised an exception)
                    future.result()
            except Exception as exc:
                # Handle the exception
                print(f"Function call with argument {arg} raised an exception: {exc}")
                # Raise an error or take appropriate action
                raise RuntimeError("One or more executor failed") from exc
            finally:
                # Ensure executor and associated processes are properly terminated
                executor.shutdown()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down gracefully...")
        executor.shutdown(wait=False, cancel_futures=True)
        logger.info(
            "Executor shut down. Kill running futures using\n"
            "'ps -ef | grep 'fd-shifts report_bootstrap' | grep -v grep | awk '{print $2}' | "
            "xargs -r kill -9'"
        )
        raise
