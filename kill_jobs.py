import subprocess

start_id = 6241084
end_id = 6241116
for id in range(start_id, end_id + 1):
    print(id)
    subprocess.call("bkill {}".format(id), shell=True)