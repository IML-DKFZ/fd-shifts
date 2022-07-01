from itertools import product

import pandas as pd

from fd_shifts.reporting import filter_best_hparams


def select_models(df):
    def select_func(row, selection_df, selection_column):
        if "openset" in row["study"]:
            return 1
        name_splitter = -1 if selection_column == "rew" else -2
        row_exp = row["study"].split("_")[0] + "_"
        row_confid = "_".join(row["confid"].split("_")[:name_splitter])
        selection_df = selection_df[
            (selection_df.study.str.contains(row_exp))
            & (selection_df.confid == row_confid)
        ]
        try:
            if row[selection_column] == selection_df[selection_column].tolist()[0]:
                return 1
            else:
                return 0
        except IndexError as e:
            print(row_exp, row_confid, len(selection_df))
            raise e

    ms_metric = "aurc"  # Careful, when changing consider changing idxmin -> idxmax

    # REWARD
    non_agg_columns = ["study", "confid", "rew"]
    ms_filter_metrics_df = df[["study", "confid", "run", "rew", ms_metric]]
    df_ms = ms_filter_metrics_df.groupby(by=non_agg_columns).mean().reset_index()
    #     print(len(df_ms), len(ms_filter_metrics_df))
    df_ms = df_ms[df_ms.study.str.contains("val_tuning")]
    df_ms["confid"] = df_ms.apply(
        lambda row: "_".join(row["confid"].split("_")[:-1]), axis=1
    )
    df_ms = df_ms.loc[
        df_ms.groupby(["study", "confid"])[ms_metric].idxmin().reset_index()[ms_metric]
    ]
    #     print(len(df), len(df_ms))
    df["select_rew"] = df.apply(lambda row: select_func(row, df_ms, "rew"), axis=1)
    selected_df = df[df.select_rew == 1]

    # DROPOUT
    non_agg_columns = ["study", "confid", "dropout"]
    # selected_df["dropout"] = selected_df.apply(
    #     lambda row: row["name"].split("do")[1].split("_")[0], axis=1
    # )
    do_filter_metrics_df = selected_df[["study", "confid", "run", "dropout", ms_metric]]
    df_do = do_filter_metrics_df.groupby(by=non_agg_columns).mean().reset_index()
    #     print(len(df_do), len(do_filter_metrics_df))
    df_do = df_do[df_do.study.str.contains("val_tuning")]
    df_do["confid"] = df_do.apply(
        lambda row: "_".join(row["confid"].split("_")[:-2]), axis=1
    )
    df_do = df_do.loc[
        df_do.groupby(["study", "confid"])[ms_metric].idxmin().reset_index()[ms_metric]
    ]
    #     print(len(df), len(selected_df), len(df_do))
    selected_df["select_do"] = selected_df.apply(
        lambda row: select_func(row, df_do, "dropout"), axis=1
    )
    all_selected_df = selected_df[selected_df.select_do == 1]
    return all_selected_df


data = pd.DataFrame.from_dict(
    {
        "study": [],
        "confid": [],
        "_confid": [],
        "experiment": [],
        "model": [],
        "rew": [],
        "dropout": [],
        "run": [],
        "aurc": [],
        "_i": [],
        "_j": [],
        "_k": [],
    }
)

studies = ["val_tuning", "study1"]
confids = [f"confid{i}" for i in range(3)]
experiments = [f"experiment{i}" for i in range(3)] + [
    f"experimentvit{i}" for i in range(3)
]
models = [f"model{i}" for i in range(3)]
rewards = list(range(3))
dropouts = [0, 1]
runs = list(range(5))

for i, (study, confid, experiment, model, reward, dropout, run) in enumerate(
    product(studies, confids, experiments, models, rewards, dropouts, runs)
):
    if (
        (experiment == f"experiment{reward}")
        or (experiment == f"experimentvit{reward}")
        or (confid == f"confid{reward}")
        # or (model == f"model{reward}")
    ) and (reward % 2 == dropout):
        aurc = reward + (3 if "vit" in experiment else 0)
    else:
        aurc = 1000

    data.loc[i] = [
        f"{experiment}_{study}",
        f"{model}_{confid}_{dropout}_{reward}",
        confid,
        experiment,
        model,
        reward,
        dropout,
        run,
        aurc,
        int(experiment[-1]),
        int(confid[-1]),
        # int(model[-1]),
        1000
    ]


expected = data[
    (
        (data.aurc == data[["_i", "_j", "_k"]].min(axis=1))
        | (data.aurc == (3 + data[["_i", "_j", "_k"]].min(axis=1)))
    )
    & (data.rew % 2 == data.dropout)
]
expected = expected.drop(["_i", "_j", "_k"], axis=1)

data = data.drop(["_i", "_j", "_k"], axis=1)
assert data is not None


def test_filter_best_hparams(capsys):
    pd.set_option("display.max_rows", None)
    selected_new = filter_best_hparams(data)
    selected_old = select_models(data).drop(["select_rew", "select_do"], axis=1)

    with capsys.disabled():
        print()
        print(
            expected.merge(selected_new, indicator=True, how="outer")
            .loc[lambda x: x["_merge"] != "both"]
            .sort_values(list(expected.columns) + ["_merge"])
        )

    pd.testing.assert_frame_equal(selected_old, expected)
    pd.testing.assert_frame_equal(selected_new, expected)
