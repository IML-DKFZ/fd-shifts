runs = [1, 2, 3]

datasets = ["dermoscopy", "chest_xray", "rxrx1", "lidc_idri"]

corruptions = ["brhigh", "brhighhigh", "brlow", "brlowlow", "gaunoilow", "gaunoilowlow"]
x_ray_corruption = [
    "brhigh",
    "brhighhigh",
    "brlow",
    "brlowlow",
    "gaunoilow",
    "gaunoilowlow",
    "letter",
]
dropout = 1
models = ["deepgamblers", "cross_entropy"]
for run in runs[0:1]:
    for dataset in datasets:
        for model in models:
            study = model + "_" + dataset
            if dataset == "dermoscopy":
                iid_all = ["dermoscopyall"]

            if dataset == "chest_xray":
                iid_all = ["xray_chestall"]
            if dataset == "rxrx1":
                iid_all = ["rxrx1all"]
            if dataset == "lidc_idri":
                iid_all = ["lidc_idriall"]
            trainings = []
            trainings.extend(iid_all)
            for training in trainings:
                if training == "dermoscopyall":
                    data = training + "_data"
                    in_class_study_ls = [
                        "dermoscopyallcorrbrhigh",
                        "dermoscopyallcorrbrhighhigh",
                        "dermoscopyallcorrbrlow",
                        "dermoscopyallcorrbrlowlow",
                        "dermoscopyallcorrgaunoilow",
                        "dermoscopyallcorrgaunoilowlow",
                    ]
                    in_class_study = f"\[{in_class_study_ls[0]}\]"

                if training == "xray_chestall":
                    data = "xray_chestall"
                    in_class_study_ls = [
                        "xray_chestallcorrbrhigh",
                        "xray_chestallcorrbrhighhigh",
                        "xray_chestallcorrbrlow",
                        "xray_chestallcorrbrlowlow",
                        "xray_chestallcorrgaunoilow",
                        "xray_chestallcorrgaunoilowlow",
                        "xray_chestallletter",
                    ]
                    in_class_study = f"\[{in_class_study_ls[0]}\]"

                if training == "rxrx1all":
                    data = "rxrx1all_data"
                    in_class_study_ls = [
                        "rxrx1all_corrbrhigh",
                        "rxrx1all_corrbrhighhigh",
                        "rxrx1all_corrbrlow",
                        "rxrx1all_corrbrlowlow",
                        "rxrx1all_corrgaunoilow",
                        "rxrx1all_corrgaunoilowlow",
                    ]
                    in_class_study = f"\[{in_class_study_ls[0]}\]"

                if training == "lidc_idriall":
                    data = "lidc_idriall_data"
                    in_class_study_ls = [
                        "lidc_idriall_corrbrhigh",
                        "lidc_idriall_corrbrhighhigh",
                        "lidc_idriall_corrbrlow",
                        "lidc_idriall_corrbrlowlow",
                        "lidc_idriall_corrgaunoilow",
                        "lidc_idriall_corrgaunoilowlow",
                    ]
                    in_class_study = f"\[{in_class_study_ls[0]}\]"  # to create
                if training == "spiculation":
                    data = "lidc_idriall_spiculation_iid_data"
                    in_class_study = f"\[lidc_idriall_spiculation_ood\]"

                if dropout == 1:
                    confidence_measures_ls = [
                        "det_mcp",
                        "det_pe",
                        "ext",
                        "mcd_mcp",
                        "mcd_pe",
                        "mcd_ee",
                    ]
                    confidence_measures = f"\[{confidence_measures_ls[0]},{confidence_measures_ls[1]},{confidence_measures_ls[2]},{confidence_measures_ls[3]},{confidence_measures_ls[4]},{confidence_measures_ls[5]}\]"

                exp_name = model + "_mcd"
                exp_group_name = f"{training}_run_{run}"
                if model == "deepgamblers":
                    for reward in [2.2, 6, 10, 20]:
                        exp_name_2 = f"{exp_name}_dg_rw_{reward}"
                        fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name_2} eval.query_studies.in_class_study={in_class_study} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures} model.dg_reward={reward}"
                        subcommand = f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -R "select[hname!=\'e230-dgxa100-2\']" -g /master_levin -q gpu-lowprio "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
                        # print(subcommand)
                        print(fd_shifts_command)
                if model == "cross_entropy" and dataset == "rxrx1":
                    for lr in [0.000015, 0.00015, 0.0015]:
                        exp_name_2 = f"{exp_name}_lr_{lr}"
                        fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name_2} eval.query_studies.in_class_study={in_class_study} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures} optimizer.learning_rate={lr}"
                        subcommand = f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -R "select[hname!=\'e230-dgxa100-2\']" -g /master_levin -q gpu-lowprio "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
                        # print(subcommand)
                        print(fd_shifts_command)

                    # os.system(subcommand)
