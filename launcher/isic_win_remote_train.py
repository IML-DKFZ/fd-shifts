import paramiko
import time

ssh = paramiko.SSHClient()
keyfilename = "/home/l049e/.ssh/id_rsa"
password = input("password: ")
k = paramiko.RSAKey.from_private_key_file(keyfilename, password=password)
# OR k = paramiko.DSSKey.from_private_key_file(keyfilename)
# hostname = "odcf-worker01"#
hostname = "bsub01.lsf.dkfz.de"
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=hostname, username="l049e", pkey=k)
study = "isic_study"
data = "isic_winner_data"
exp_group_name = "isic_winner"
exp_name = "effnetb4"
numEpochs = "15"
batchsize = "64"
fd_shifts_command = f"fd_shifts study={study} data={data} exp.group_name={exp_group_name} exp.name={exp_name} eval.query_studies.iid_study={data} trainer.batch_size={batchsize} trainer.num_epochs={numEpochs} trainer.do_val=True"
subcommand = f'bsub -gpu num=8j_exclusive=yes:mode=exclusive_process:gmem=20G -L /bin/bash -q gpu "source ~/.bashrc && conda activate fd-shifts && {fd_shifts_command}"'
print(subcommand)
stdin_, stdout_, stderr_ = ssh.exec_command(subcommand)
# stdin_, stdout_, stderr_ = ssh.exec_command(f"ls -l ~")
# time.sleep(2)    # Previously, I had to sleep for some time.
# stdout_.channel.recv_exit_status()
# lines = stdout_.readlines()
# for line in lines:
#    print(line)
#
ssh.close()
