# import paramiko
import time
import getpass
import os

# ssh = paramiko.SSHClient()
# keyfilename = "/home/l049e/.ssh/id_rsa"
# password = getpass.getpass("Password: ")
# k = paramiko.RSAKey.from_private_key_file(keyfilename, password=password)
## OR k = paramiko.DSSKey.from_private_key_file(keyfilename)
## hostname = "odcf-worker01"#
# hostname = "bsub01.lsf.dkfz.de"
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(hostname=hostname, username="l049e", pkey=k)
study = "confidnet_dermoscopy"
data_ls = [
    "dermoscopyall_data",
    "dermoscopyallbutbarcelona_data",
    "dermoscopyallbutd7p_data",
    "dermoscopyallbutmskcc_data",
    "dermoscopyallbutpascal_data",
    "dermoscopyallbutph2_data",
    "dermoscopyallbutqueensland_data",
    "dermoscopyallbutvienna_data",
    "ham10000subbig_data",
]
lr = 0.003
for data in data_ls:
    for dropout in [0, 1]:
        start, _ = data.split("_data")
        exp_group_name = f"{start}_run1"
        accelerator = "None"
        exp_name = "confidnet"
        num_epochs = "30"
        batchsize = "12"
        if data == "dermoscopyall_data":
            in_class_study_ls = [
                "dermoscopyallcorrbrhigh",
                "dermoscopyallcorrbrhighhigh",
                "dermoscopyallcorrbrlow",
                "dermoscopyallcorrbrlowlow",
                "dermoscopyallcorrgaunoilow",
                "dermoscopyallcorrgaunoilowlow",
                "dermoscopyallcorrelastichigh",
                "dermoscopyallcorrelastichighhigh",
                "dermoscopyallcorrmotblrhigh",
                "dermoscopyallcorrmotblrhighhigh",
            ]
            in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]},{in_class_study_ls[6]},{in_class_study_ls[7]},{in_class_study_ls[8]},{in_class_study_ls[9]}\]"
        elif data == "ham10000subbig_data":
            in_class_study = f"\[ham10000subsmall\]"
            lr = 0.0003
        else:
            start, end = data.split("but")
            attribution, _ = end.split("_data")
            in_class_study_ls = start + attribution
            in_class_study = f"\[{in_class_study_ls}\]"

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
            exp_name = exp_name + "_mcd"

            exp_name = exp_name + f"_lr{lr}"

            fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.in_class_study={in_class_study} trainer.accelerator={accelerator} trainer.learning_rate={lr} trainer.batch_size={batchsize} trainer.num_epochs={num_epochs} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"
            subcommand = f'bsub -gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -q gpu-lowprio "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
            # print(subcommand)
            print(fd_shifts_command)
            # os.system(subcommand)
            # os.system(subcommand)
# subcommand = "echo $PATH"
# channel = ssh.get_transport().open_session()
#
# channel.get_pty()
# channel.invoke_shell()
# print(channel.recv(10024))
# channel.send("ls")
# time.sleep(5)
# print(channel.recv(10024))

# stdin_, stdout_, stderr_ = ssh.exec_command(subcommand)
## stdin_, stdout_, stderr_ = ssh.exec_command(f"git pull")
# time.sleep(2)  # Previously, I had to sleep for some time.
# print(stdout_.read().decode())
# print(stderr_.read().decode())

# stdin2_, stdout2_, stderr2_ = ssh.exec_command(subcommand)
# stdin_, stdout_, stderr_ = ssh.exec_command(f"git pull")
# time.sleep(2)  # Previously, I had to sleep for some time.
# print(stdout2_.read().decode())
# stdout_.channel.recv_exit_status()
# lines = stdout_.readlines()
# for line in lines:
#    print(line)
# ssh.close()
