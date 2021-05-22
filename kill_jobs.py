import subprocess

start_id = 6292945
end_id = 6292984
for id in range(start_id, end_id + 1):
    print(id)
    subprocess.call("bkill {}".format(id), shell=True)