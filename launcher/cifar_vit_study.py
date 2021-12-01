import os
import subprocess
import time
from itertools import product
from pathlib import Path

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")

cn_pretrained_bbs = {
    "cifar10": [
        "/gpu/checkpoints/OE0612/t974t/experiments/vit/cifar10_lr0.0003_run0/version_0/last.ckpt",
        "/gpu/checkpoints/OE0612/t974t/experiments/vit/cifar10_lr0.01_run0_do1/version_0/last.ckpt",
    ],
    "cifar100": [
        "",
        "",
    ],
    "super_cifar100": [
        "",
        "",
    ],
    "svhn": [
        "",
        "",
    ],
    "breeds": [
        "",
        "",
    ],
    "wilds_animals": [
        "",
        "",
    ],
    "wilds_camelyon": [
        "",
        "",
    ],
}

rewards = [2.2, 3, 4.5, 6, 10]
experiments: list[tuple[list, list, list, list, list, list, list, range]] = [
    (["cifar10"], ["confidnet"], ["vit"], [0.01], [128], [1], [2.2], range(1)),
    # (["breeds"], ["dg"], ["vit"], [0.01], [128], [1], rewards, range(1)),
    # (["svhn"], ["dg"], ["vit"], [0.01], [128], [1], rewards, range(1)),
    # (["wilds_camelyon"], ["dg"], ["vit"], [0.003], [128], [1], rewards, range(1)),
    # (["cifar10"], ["devries"], ["vit"], [0.0003], [128], [0], [2.2], range(1)),
    # (["cifar10"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(1)),
    # (["breeds"], ["devries"], ["vit"], [0.001], [128], [0], [2.2], range(1)),
    # (["breeds"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(1)),
    # (["svhn"], ["devries"], ["vit"], [0.01], [128], [0], [2.2], range(1)),
    # (["svhn"], ["devries"], ["vit"], [0.01], [128], [1], [2.2], range(1)),
    # (["wilds_camelyon"], ["devries"], ["vit"], [0.001], [128], [0], [2.2], range(1)),
    # (["wilds_camelyon"], ["devries"], ["vit"], [0.003], [128], [1], [2.2], range(1)),
    # (["cifar100"], ["dg"], ["vit"], [0.01], [512], [1], rewards, range(1)),
    # (["wilds_animals"], ["dg"], ["vit"], [0.01], [512], [1], rewards, range(1)),
    # (["cifar100"], ["devries"], ["vit"], [0.03], [512], [0], [2.2], range(1)),
    # (["wilds_animals"], ["devries"], ["vit"], [0.001], [512], [0], [2.2], range(1)),
    # (["cifar100"], ["devries"], ["vit"], [0.01], [512], [1], [2.2], range(1)),
    # (["wilds_animals"], ["devries"], ["vit"], [0.01], [512], [1], [2.2], range(1)),
    # (["super_cifar100"], ["dg"], ["vit"], [0.001], [128], [1], rewards, range(1)),
    # (["super_cifar100"], ["devries"], ["vit"], [0.003], [128], [0], [2.2], range(1)),
    # (["super_cifar100"], ["devries"], ["vit"], [0.001], [128], [1], [2.2], range(1)),
]

for experiment in experiments:
    for dataset, model, bb, lr, bs, do, rew, run in product(*experiment):
        exp_name = "{}_model{}_bb{}_lr{}_bs{}_run{}_do{}_rew{}".format(
            dataset,
            model,
            bb,
            lr,
            bs,
            run,
            do,
            rew,
        )

        if dataset in ["cifar100", "wilds_animals"]:
            base_command = """bsub \\
-R "select[hname!='e230-dgx2-1']" \\
-gpu num=16:j_exclusive=yes:mode=exclusive_process:gmem=31.7G \\
-L /bin/bash -q gpu-lowprio \\
-g /t974t/train_big \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
-J "{}" \\\n"""
        elif model == "confidnet":
            base_command = """bsub \\
-R "select[hname!='e230-dgx2-1']" \\
-gpu num=4:j_exclusive=yes:gmem=31.7G \\
-L /bin/bash -q gpu-lowprio \\
-g /t974t/train_small \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
-J "{}" \\\n"""
        else:
            base_command = """bsub \\
-R "select[hname!='e230-dgx2-1']" \\
-gpu num=4:j_exclusive=yes:gmem=31.7G \\
-L /bin/bash -q gpu-lowprio \\
-g /t974t/train \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
-J "{}" \\\n"""

        base_command = (
            base_command
            + """'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}'"""
        )

        # base_command = "echo {} && bash -li -c 'source ~/.bashrc && conda activate failure-detection && EXPERIMENT_ROOT_DIR=/home/t974t/cluster/experiments DATASET_ROOT_DIR=/home/t974t/Data python -W ignore {} {}'"

        command_line_args = [
            "exp.mode={}".format("train"),
            "trainer.batch_size={}".format(bs),
            "+trainer.accelerator=ddp",
            "+model.dropout_rate={}".format(do),
            "study={}_vit_study".format(dataset),
            "exp.name={}".format(exp_name),
            "trainer.learning_rate={}".format(lr),
            "trainer.val_split=devries",
            "+trainer.do_val=true",
            "+eval.val_tuning=true",
            "+eval.r_star=0.25",
            "+eval.r_delta=0.05",
        ]

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
                    "++model.network.save_dg_backbone_path='\"'\"'${exp.dir}/dg_backbone.ckpt'\"'\"'",
                    "++eval.ext_confid_name=dg",
                    "++eval.test_conf_scaling=false",
                    "++test.iid_set_split=devries",
                ]
            )
            if do == 1:
                command_line_args.append("++model.avg_pool={}".format(False))

        if model == "confidnet":
            pretrained_path = Path(cn_pretrained_bbs[dataset][do]).expanduser()
            # TODO: Check epochs and milestones when loading pretrained backbone -> do this in training_stages.py?
            command_line_args.extend(
                [
                    "++trainer.num_epochs=220",
                    "~trainer.num_steps",
                    "++trainer.lr_scheduler.name=LinearWarmupCosineAnnealing",
                    "++trainer.lr_scheduler.warmup_epochs=0",
                    "++trainer.lr_scheduler.max_epochs='\"'\"'${trainer.num_epochs_backbone}'\"'\"'",
                    "++trainer.weight_decay=0.0005",
                    "++model.name=confidnet_model",
                    "++model.network=None",
                    "++test.selection_mode=null",
                    "++test.iid_set_split=devries",
                    "++eval.ext_confid_name=tcp",
                    "++trainer.num_epochs_backbone=0",
                    "++trainer.learning_rate_confidnet=1e-4",
                    "++trainer.learning_rate_confidnet_finetune=1e-6",
                    '++trainer.callbacks.training_stages.milestones="[0, 200]"',
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

        launch_command = base_command.format(
            exp_name, exec_path, " ".join(command_line_args)
        )

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)
        time.sleep(1)

        # TESTING

        base_command = """bsub \\
        -gpu num=4:j_exclusive=yes:gmem=10.7G \\
        -L /bin/bash -q gpu-lowprio \\
        -u 'till.bungert@dkfz-heidelberg.de' -B -N \\
        -w "done({})" \\
        -g /t974t/test \\
        -J "{}_test" \\
        'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}\'"""
        # base_command = "echo {} && bash -li -c 'source ~/.bashrc && conda activate failure-detection && EXPERIMENT_ROOT_DIR=/home/t974t/cluster/experiments DATASET_ROOT_DIR=/home/t974t/Data python -W ignore {} {}'"

        command_line_args[0] = "exp.mode={}".format("test")
        command_line_args[1] = "trainer.batch_size=128"
        command_line_args[2] = "+trainer.accelerator=ddp"
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

        launch_command = base_command.format(
            exp_name,
            exp_name,
            exec_path,
            " ".join(command_line_args),
        )

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)
        time.sleep(1)
