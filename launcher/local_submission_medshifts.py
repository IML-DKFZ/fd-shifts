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
models = ["devries", "confidnet", "deepgamblers"]
for run in runs[0:1]:
    for dataset in datasets:
        for model in models:
            study = model + "_" + dataset
            if dataset == "dermoscopy":
                center_shifts = ["barcelona", "mskcc"]
                sub_class_shifts = ["ham10000subclass"]
                iid_all = ["dermoscopyall"]
                bottom_up = ["ham10000multi", "mskcc"]
            if dataset == "chest_xray":
                center_shifts = ["butnih14", "butchexpert"]
                sub_class_shifts = []
                iid_all = ["xray_chestall"]
                bottom_up = ["nih14"]
            if dataset == "rxrx1":
                center_shifts = []
                sub_class_shifts = ["HEPG2vs3", "40vs11set1"]
                iid_all = ["rxrx1all"]
                bottom_up = []
            if dataset == "lidc_idri":
                center_shifts = []
                sub_class_shifts = ["spiculation", "texture", "calcification"]
                iid_all = ["lidc_idriall"]
                bottom_up = []
            trainings = []
            trainings.extend(iid_all)
            trainings.extend(center_shifts)
            trainings.extend(sub_class_shifts)

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
                    in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]}\]"
                if training == "barcelona":
                    data = "dermoscopyallbutbarcelona_data"
                    in_class_study = f"\[dermoscopyallbarcelona\]"
                if training == "mskcc":
                    data = "dermoscopyallbutmskcc_data"
                    in_class_study = f"\[dermoscopyallmskcc\]"
                if training == "ham10000subclass":
                    data = "dermoscopyallham10000subbig_data"
                    in_class_study = f"\[dermoscopyallham10000subsmall\]"

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
                    in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]},{in_class_study_ls[6]}\]"
                if training == "butnih14":
                    data = "xray_chestallbutnih14_data"
                    in_class_study = f"\[xray_chestallnih14\]"
                if training == "butchexpert":
                    data = "xray_chestallbutchexpert_data"
                    in_class_study = f"\[xray_chestallchexpert\]"
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
                    in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]}\]"
                if training == "HEPG2vs3":
                    data = "rxrx1allbuthepg2_data"  # to create
                    in_class_study = f"\[rxrx1allhepg2\]"  # to create
                if training == "40vs11set1":
                    data = "rxrx1all40sset1_data"
                    in_class_study = f"\[rxrx1all10sset1\]"
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
                    in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]}\]"  # to create
                if training == "spiculation":
                    data = "lidc_idriall_spiculation_iid_data"
                    in_class_study = f"\[lidc_idriall_spiculation_ood\]"
                if training == "texture":
                    data = "lidc_idriall_texture_iid_data"
                    in_class_study = f"\[lidc_idriall_texture_ood\]"
                if training == "calcification":
                    data = "lidc_idriall_calcification_iid_data"
                    in_class_study = f"\[lidc_idriall_calcification_ood\]"

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
                fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.in_class_study={in_class_study} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"
                subcommand = f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -R "select[hname!=\'e230-dgxa100-2\']" -g /master_levin -q gpu-lowprio "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
                # print(subcommand)
                print(fd_shifts_command)

                # os.system(subcommand)
