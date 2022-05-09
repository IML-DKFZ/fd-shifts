import subprocess

for lr in [1, 2, 3, 4]:
    for dropout in [0, 1]:
        # print(f"{lr} and {dropout}")
        learning_rate = 10 ** (-lr)
        # subprocess.run([f"fd_shifts data=emnist_data_balanced exp.group_name=test_emnist_hp_optim exp.name=emnist_balanced_hp_lr_e{lr}_drO_{dropout} eval.query_studies.iid_study=emnist_balanced eval.query_studies.new_class_study=[emnist_balanced] trainer.num_epochs=1 trainer.optimizer.learning_rate={learning_rate} model.dropout_rate={dropout}"], shell=True)
        subprocess.run(
            [
                f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G -L /bin/bash -q gpu "source ~/.bashrc && conda activate fd-shifts && fd_shifts data=med_mnist_path exp.group_name=hp_grid_med_mnist_path exp.name=med_mnist_path_hp_lr_e{lr}_drO_{dropout} eval.query_studies.iid_study=med_mnist_path eval.query_studies.new_class_study=[emnist_balanced] trainer.num_epochs=200 trainer.optimizer.learning_rate={learning_rate} model.dropout_rate={dropout} trainer.optimizer.weight_decay=0.01"'
            ],
            shell=True,
        )
