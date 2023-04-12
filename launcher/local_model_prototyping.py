# import paramiko
import getpass
import os
import time

studies = [
    "confidnet_chest_xray",
    "cross_entropy_chest_xray",
    "deepgamblers_chest_xray",
    "devries_chest_xray",
    "confidnet_dermoscopy",
    "cross_entropy_dermoscopy",
    "deepgamblers_dermoscopy",
    "devries_dermoscopy",
    "confidnet_lidc_lidr",
    "cross_entropy_lidc_lidr",
    "deepgamblers_lidc_lidr",
    "devries_lidc_lidr",
    "confidnet_rxrx1",
    "cross_entropy_rxrx1",
    "deepgamblers_rxrx1",
    "devries_rxrx1",
]
for study in studies:
    if "rxrx1" in study:
        data = "rxrx1all_data"
    if "chest_xray" in study:
        data = "xray_chestall_data"
    if "dermoscopy" in study:
        data = "dermoscopyall_data"
    if "lidc_idri" in study:
        data = "lidc_idriall_data"

    for dropout in [0, 1]:
        start, _ = data.split("_data")
        exp_group_name = f"{start}_proto"
        accelerator = "None"
        exp_name = "det"
        num_epochs = 1
        attribution, _ = data.split("_data")
        in_class_study = f"\[{attribution}\]"

        confidence_measures_ls = ["det_mcp", "det_pe", "ext"]
        confidence_measures = f"\[{confidence_measures_ls[0]},{confidence_measures_ls[1]},{confidence_measures_ls[2]}\]"

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
            exp_name = "mcd"

        fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.in_class_study={in_class_study} trainer.accelerator={accelerator} trainer.num_epochs={num_epochs} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"
        subcommand = f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -R "select[hname!=\'e230-dgxa100-2\']" -g /master_levin -q gpu-lowprio "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
        # print(subcommand)
        print(fd_shifts_command)
        # os.system(subcommand)source ~/.bashrc && conda activate fd-shifts && fd_shifts study=cross_entropy_dermoscopy data=dermoscopyallham10000_data exp.group_name=dermoscopyallham10000_ce_run1 exp.name=ce_mcd eval.query_studies.in_class_study=\[dermoscopyallham10000\] trainer.accelerator=None trainer.learning_rate=3e-05 trainer.batch_size=12 trainer.num_epochs=40 model.dropout_rate=1 eval.confidence_measures.test=\[det_mcp,det_pe,mcd_mcp,mcd_pe,mcd_ee\]
        # os.system(subcommand)
