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
study = "devries_mod_xray"
data_ls = [
    "xray_chestall_data",
    "xray_chestallbutnih14_data",
    "xray_chestallbutchexpert_data",
    "xray_chestallbutmimic_data",
]
for data in data_ls[0:1]:
    dropout = 0
    start, _ = data.split("_data")
    exp_group_name = f"{start}_run1"
    accelerator = "None"
    exp_name = "devries"
    num_epochs = "120"
    batchsize = "48"
    if data == "xray_chestall_data":
        in_class_study_ls = [
            "xray_chestallcorrletter",
            "xray_chestallcorrbrhigh",
            "xray_chestallcorrbrhighhigh",
            "xray_chestallcorrbrlow",
            "xray_chestallcorrbrlowlow",
            "xray_chestallcorrgaunoihigh",
            "xray_chestallcorrgaunoihighhigh",
            "xray_chestallcorrelastichigh",
            "xray_chestallcorrelastichighhigh",
            "xray_chestallcorrmotblrhigh",
            "xray_chestallcorrmotblrhighhigh",
        ]
        in_class_study = f"\[{in_class_study_ls[0]},{in_class_study_ls[1]},{in_class_study_ls[2]},{in_class_study_ls[3]},{in_class_study_ls[4]},{in_class_study_ls[5]},{in_class_study_ls[6]},{in_class_study_ls[7]},{in_class_study_ls[8]},{in_class_study_ls[9]}\]"
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

    # fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.in_class_study={in_class_study} trainer.accelerator={accelerator} trainer.batch_size={batchsize} trainer.num_epochs={num_epochs} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"

    fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} trainer.accelerator={accelerator} trainer.batch_size={batchsize} trainer.num_epochs={num_epochs} model.dropout_rate={dropout} eval.confidence_measures.test={confidence_measures}"

    subcommand = f'bsub -gpu num=2:j_exclusive=yes:mode=exclusive_process:gmem=22G -L /bin/bash -q gpu "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
    print(fd_shifts_command)
    os.system(fd_shifts_command)
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
