import os
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Optional, Union

system_name = os.environ["SYSTEM_NAME"]

if system_name == "cluster":
    base_path = Path("/gpu/checkpoints/OE0612/t974t/experiments/")
elif system_name == "local":
    base_path = Path("~/cluster/experiments/").expanduser()
else:
    raise ValueError(
        "Environment Variable SYSTEM_NAME must be either 'cluster' or 'local'"
    )


def get_base_command(mode: str, model: str, dataset: str, stage: Optional[int]):
    if system_name == "local":
        return (
            "echo {exp_name} && bash -li -c 'source ~/.bashrc && conda activate failure-detection "
            "&& EXPERIMENT_ROOT_DIR=/home/t974t/cluster/experiments "
            "DATASET_ROOT_DIR=/home/t974t/Data "
            "python -W ignore {cmd} {args}'"
        )

    if mode == "test":
        return " \\\n".join(
            [
                "bsub",
                "-gpu num=4:j_exclusive=yes:gmem=10.7G",
                "-L /bin/bash -q gpu-lowprio",
                "-u 'till.bungert@dkfz-heidelberg.de' -B -N",
                '-w "done({exp_name})"',
                "-g /t974t/test",
                '-J "{exp_name}_test"',
                (
                    "'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && "
                    "python -W ignore {cmd} {args}'"
                ),
            ]
        )

    base_command = [
        "bsub",
        "-R \"select[hname!='e230-dgx2-1']\"",
    ]

    # if dataset in ["cifar100", "wilds_animals"]:
    #     base_command.extend(
    #         [
    #             "-gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G",
    #             '-J "{exp_name}"',
    #             "-g /t974t/train",
    #         ]
    #     )
    if model == "confidnet" and stage == 1:
        base_command.extend(
            [
                "-gpu num=4:j_exclusive=yes:gmem=10.7G",
                "-g /t974t/train_small",
                '-J "{exp_name}"',
            ]
        )
    elif model == "confidnet" and stage == 2:
        base_command.extend(
            [
                "-gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G",
                "-g /t974t/train",
                '-w "ended({exp_name})"',
                '-J "{exp_name}_stage2"',
            ]
        )
    else:
        base_command.extend(
            [
                "-gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G",
                "-g /t974t/train",
                '-J "{exp_name}"',
            ]
        )

    base_command.extend(
        [
            "-L /bin/bash -q gpu",
            "-u 'till.bungert@dkfz-heidelberg.de' -B -N",
            "'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {cmd} {args}'",
        ]
    )

    return " \\\n".join(base_command)


def check_done():
    pass


current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")

cn_pretrained_bbs = {
    "cifar10": [
        "vit/cifar10_lr0.0003_run0/version_0/last.ckpt",
        "vit/cifar10_lr0.0003_run0/version_0/last.ckpt",
    ],
    "cifar100": [
        "vit/cifar100_lr0.03_run0/version_4/last.ckpt",
        "vit/cifar100_lr0.01_run0_do1/version_0/last.ckpt",
    ],
    "super_cifar100": [
        "vit/super_cifar100_lr0.003_run0_do0/version_0/last.ckpt",
        "vit/super_cifar100_lr0.001_run1_do1/version_0/last.ckpt",
    ],
    "svhn": [
        "vit/svhn_lr0.01_run0/version_0/last.ckpt",
        "vit/svhn_lr0.01_run0_do1/version_0/last.ckpt",
    ],
    "breeds": [
        "vit/breeds_lr0.001_run0/version_2/last.ckpt",
        "vit/breeds_lr0.01_run0_do1/version_0/last.ckpt",
    ],
    "wilds_animals": [
        "vit/wilds_animals_lr0.001_run0/version_2/last.ckpt",
        "vit/wilds_animals_lr0.01_run0_do1/version_0/last.ckpt",
    ],
    "wilds_camelyon": [
        "vit/wilds_camelyon_lr0.001_run0_do0/version_1/last.ckpt",
        "vit/wilds_camelyon_lr0.003_run0_do1/version_0/last.ckpt",
    ],
}

MODE = "train_test"

rewards = [2.2, 3, 4.5, 6, 10]
experiments: list[
    tuple[list, list, list, list, list, list, list, Union[range, list], Optional[list]]
] = [
    # (
    #     ["super_cifar100"],
    #     ["confidnet"],
    #     ["vit"],
    #     [0.001],
    #     [128],
    #     [1],
    #     [3],
    #     range(1),
    #     [1, 2],
    # ),
    # (["cifar100"], ["confidnet"], ["vit"], [0.01], [128], [1], [2.2], [2, 4], [1, 2],),
    # (
    #     ["wilds_animals"],
    #     ["confidnet"],
    #     ["vit"],
    #     [0.01],
    #     [128],
    #     [1],
    #     [2.2],
    #     [3],
    #     [1, 2],
    # ),
    # (["svhn_openset"], ["confidnet"], ["vit"], [0.01], [128], [1], [2.2], range(0, 5), [1, 2]),
    # (["svhn_openset"], ["devries"], ["vit"], [0.01], [128], [0], [2.2], range(1, 5), [None]),
    # (["svhn_openset"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(0, 5), [None]),
    # (["svhn_openset"], ["dg"], ["vit"], [0.01], [128], [1], [4.5], range(0, 5), [None]),
    (["svhn_openset"], ["vit"], ["vit"], [0.01], [128], [0], [0], range(0, 5), [None]),
    (["svhn_openset"], ["vit"], ["vit"], [0.01], [128], [1], [0], range(0, 5), [None]),
    # (["wilds_animals_openset"], ["confidnet"], ["vit"], [0.01], [128], [1], [2.2], range(0, 5), [1, 2]),
    # (["wilds_animals_openset"], ["devries"], ["vit"], [0.001], [128], [0], [2.2], range(0, 5), [None]),
    # (["wilds_animals_openset"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(0, 5), [None]),
    # (["wilds_animals_openset"], ["dg"], ["vit"], [0.01], [128], [1], [10], range(0, 5), [None]),
    # (["wilds_animals_openset"], ["vit"], ["vit"], [0.001], [128], [0], [0], range(0, 5), [None]),
    # (["wilds_animals_openset"], ["vit"], ["vit"], [0.01], [128], [1], [0], range(0, 5), [None]),
]

for experiment in experiments:
    for dataset, model, bb, lr, bs, do, rew, run, stage in product(*experiment):
        exp_name = "{}_model{}_bb{}_lr{}_bs{}_run{}_do{}_rew{}".format(
            dataset, model, bb, lr, bs, run, do, rew,
        )

        command_line_args = [
            "exp.mode={}".format("train"),
            "trainer.batch_size={}".format(bs),
            "+trainer.accelerator=dp",
            "+model.dropout_rate={}".format(do),
            "study={}_vit_study".format(dataset),
            "exp.name={}".format(exp_name),
            "trainer.learning_rate={}".format(lr),
            "trainer.val_split=devries",
            "+trainer.do_val=true",
            "+eval.val_tuning=true",
            "+eval.r_star=0.25",
            "+eval.r_delta=0.05",
            "++trainer.resume_from_ckpt_confidnet=false",
        ]

        if dataset in ["cifar100", "wilds_animals"]:
            command_line_args.append("++trainer.accumulate_grad_batches=4")

        if model == "devries":
            command_line_args.extend(
                [
                    "++trainer.num_epochs=250",
                    "~trainer.num_steps",
                    "++trainer.lr_scheduler.name=CosineAnnealing",
                    "++trainer.lr_scheduler.max_epochs=250",
                    '++trainer.lr_scheduler.milestones="[60, 120, 160]"',
                    "++trainer.lr_scheduler.gamma=0.2",
                    "++trainer.optimizer.name=SGD ",
                    "++trainer.optimizer.learning_rate={}".format(lr),
                    "++trainer.optimizer.momentum=0.9",
                    "++trainer.optimizer.nesterov=true",
                    "++trainer.optimizer.weight_decay=0.0005",
                    "++model.name=devries_model",
                    "++model.fc_dim=768",
                    "++model.confidnet_fc_dim=400",
                    "++model.dg_reward=-1.0",
                    "++model.avg_pool=true",
                    "++model.monitor_mcd_samples=50",
                    "++model.test_mcd_samples=50",
                    "++model.budget=0.3",
                    "++model.network.name=devries_and_enc",
                    "++model.network.backbone={}".format(bb),
                    "++eval.ext_confid_name=devries",
                    "++eval.test_conf_scaling=false",
                    "++test.iid_set_split=devries",
                ]
            )

        if model == "dg":
            command_line_args.extend(
                [
                    "++trainer.num_epochs={}".format(300),
                    "++trainer.dg_pretrain_epochs={}".format(100),
                    "~trainer.num_steps",
                    "++trainer.lr_scheduler.name=LinearWarmupCosineAnnealing",
                    "++trainer.lr_scheduler.warmup_epochs=3",
                    "++trainer.lr_scheduler.max_epochs=300",
                    "++trainer.lr_scheduler.gamma=0.5",
                    "++trainer.optimizer.name=SGD ",
                    "++trainer.optimizer.learning_rate={}".format(lr),
                    "++trainer.optimizer.momentum=0.9",
                    "++trainer.optimizer.nesterov=false",
                    "++trainer.optimizer.weight_decay=0.0005",
                    "++model.name=devries_model",
                    "++model.fc_dim=768",
                    "++model.confidnet_fc_dim=400",
                    "++model.dg_reward={}".format(rew),
                    "++model.monitor_mcd_samples=50",
                    "++model.test_mcd_samples=50",
                    "++model.budget=0.3",
                    "++model.network.name=vit",
                    "++model.network.backbone={}".format("null"),
                    (
                        "++model.network.save_dg_backbone_path="
                        "'\"'\"'${exp.dir}/dg_backbone.ckpt'\"'\"'"
                    ),
                    "++eval.ext_confid_name=dg",
                    "++eval.test_conf_scaling=false",
                    "++test.iid_set_split=devries",
                ]
            )
            if do == 1:
                command_line_args.append("++model.avg_pool={}".format(False))

        if model == "confidnet":
            pretrained_path = base_path / cn_pretrained_bbs[dataset][do]
            if stage == 1:
                command_line_args.append("++trainer.num_epochs=220",)
                command_line_args.append(
                    '++trainer.callbacks.training_stages.milestones="[0, 200]"',
                )
                command_line_args.append("++trainer.accelerator=ddp")
                command_line_args.append("++trainer.resume_from_ckpt_confidnet=false")
                command_line_args.append("++trainer.accumulate_grad_batches=1")
            elif stage == 2:
                command_line_args.append("++trainer.num_epochs=20",)
                command_line_args.append(
                    '++trainer.callbacks.training_stages.milestones="[0, 0]"',
                )
                command_line_args.append("++trainer.batch_size=128",)
                # command_line_args.append("++trainer.accumulate_grad_batches=2")
                command_line_args.append("++trainer.resume_from_ckpt_confidnet=true")
            command_line_args.extend(
                [
                    "~trainer.num_steps",
                    "++trainer.lr_scheduler.name=LinearWarmupCosineAnnealing",
                    "++trainer.lr_scheduler.warmup_epochs=0",
                    (
                        "++trainer.lr_scheduler.max_epochs="
                        "'\"'\"'${trainer.num_epochs_backbone}'\"'\"'"
                    ),
                    "++trainer.weight_decay=0.0005",
                    "++model.name=confidnet_model",
                    "++model.network=None",
                    "++test.selection_mode=null",
                    "++test.iid_set_split=devries",
                    "++eval.ext_confid_name=tcp",
                    "++trainer.num_epochs_backbone=0",
                    "++trainer.learning_rate_confidnet=1e-4",
                    "++trainer.learning_rate_confidnet_finetune=1e-6",
                    "++trainer.callbacks.training_stages.pretrained_backbone_path={}".format(
                        pretrained_path
                    ),
                    "++trainer.callbacks.training_stages.pretrained_confidnet_path=null",
                    "++trainer.callbacks.training_stages.disable_dropout_at_finetuning=true",
                    "++trainer.callbacks.training_stages.confidnet_lr_scheduler=false",
                    "++model.fc_dim=768",
                    "++model.avg_pool=True",
                    "++model.confidnet_fc_dim=400",
                    "++model.monitor_mcd_samples=50",
                    "++model.test_mcd_samples=50",
                    "++model.network.name=confidnet_and_enc",
                    "++model.network.backbone=vit",
                    "++model.network.imagenet_weights_path=null",
                    "++data.reproduce_confidnet_splits={}".format("True"),
                ]
            )

        base_command = get_base_command("train", model, dataset, stage)

        launch_command = base_command.format(
            exp_name=exp_name, cmd=exec_path, args=" ".join(command_line_args),
        )

        if "train" in MODE:
            print("Launch command: ", launch_command, end="\n\n")
            subprocess.call(launch_command, shell=True)
            time.sleep(1)

        # TESTING

        if model == "confidnet" and stage == 1:
            continue

        command_line_args[0] = "exp.mode={}".format("test")
        command_line_args.append("++trainer.batch_size=128")
        command_line_args.append("++trainer.accelerator=ddp")
        if do == 1:
            command_line_args.append(
                'eval.confidence_measures.test="{}"'.format(
                    [
                        "det_mcp",
                        "det_pe",
                        "ext",
                        "ext_mcd",
                        "ext_waic",
                        "mcd_mcp",
                        "mcd_pe",
                        "mcd_ee",
                        "mcd_mi",
                        "mcd_sv",
                        "mcd_waic",
                    ]
                )
            )

        base_command = get_base_command("test", model, dataset, stage)

        if stage == 2:
            exp_name = exp_name + "_stage2"

        launch_command = base_command.format(
            exp_name=exp_name, cmd=exec_path, args=" ".join(command_line_args),
        )

        if "test" in MODE:
            print("Launch command: ", launch_command, end="\n\n")
            subprocess.call(launch_command, shell=True)
            time.sleep(1)
