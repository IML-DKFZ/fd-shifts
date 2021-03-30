import os
import subprocess
from itertools import product


system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")


current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

# TEST

# overrides
command_line_args = ""
command_line_args += "exp.name={} ".format("repro_mcd_mcp_2")
command_line_args += "exp.group_name={} ".format("repro_related_work")
command_line_args += "hydra.output_subdir={} ".format("null")
command_line_args += "exp.mode={} ".format("test")

# submission specs
launch_command = ""
launch_command += "bsub "
launch_command += "-gpu num=1:"
launch_command += "j_exclusive=yes:"
launch_command += "mode=exclusive_process:"
launch_command += "gmem=10.7G "
launch_command += "-L /bin/bash -q gpu "
launch_command += "'source ~/.bashrc && "
launch_command += "source ~/.virtualenvs/confid/bin/activate && "
launch_command += "python -u {} ".format(exec_path)
launch_command += command_line_args
launch_command += "'"

print("Launch command: ", launch_command)
subprocess.call(launch_command, shell=True)


#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)