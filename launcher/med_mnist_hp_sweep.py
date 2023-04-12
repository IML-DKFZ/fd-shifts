import subprocess

set = {"blood", "breast", "derma", "oct", "organ_a", "path", "pneu", "tissue"}

for name in set:
    for lr in [1, 2, 3, 4]:
        for dropout in [0, 1]:
            for wd in [1, 2, 3, 4]:
                # print(f"{lr} and {dropout}")
                learning_rate = 10 ** (-lr)
                weight_decay = 10 ** (-wd)
                # subprocess.run([f"fd_shifts data=emnist_data_balanced exp.group_name=test_emnist_hp_optim exp.name=emnist_balanced_hp_lr_e{lr}_drO_{dropout} eval.query_studies.iid_study=emnist_balanced eval.query_studies.new_class_study=[emnist_balanced] trainer.num_epochs=1 trainer.optimizer.learning_rate={learning_rate} model.dropout_rate={dropout}"], shell=True)
                data_name = "med_mnist_" + name
                group_name = "hp_grid_med_mnist" + name
                exp_name = f"{name}_lr_e{lr}_drO_{dropout}_wgdc_e{wd}"
                subprocess.run(
                    [
                        f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate fd-shifts && fd_shifts data={data_name} exp.group_name={group_name} exp.name={exp_name} eval.query_studies.iid_study={data_name} eval.query_studies.new_class_study=[emnist_balanced] trainer.num_epochs=150 trainer.optimizer.learning_rate={learning_rate} model.dropout_rate={dropout} trainer.optimizer.weight_decay={weight_decay} trainer.do_val=True"'
                    ],
                    shell=True,
                )
                # print(data_name, group_name, exp_name)
