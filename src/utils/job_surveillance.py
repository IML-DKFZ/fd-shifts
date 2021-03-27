
import argparse
import time
import subprocess
import os

def file_len(fname):
    i = -1
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
    except:
        i = 0
    return i + 1


def check_jobs():

    j_finished = []
    j_error = []

    for jid, err_path in zip(j_ids, err_files_list):
        if os.path.isfile(err_path) and file_len(err_path) > 0:
            j_error.append(jid)

    running_ids = subprocess.check_output('squeue -u pjaeger -o "%.9i %t" --noheader', shell=True)
    running_ids = str(running_ids).strip('b').split(' ')
    # print(running_ids)
    ids = [i for i in running_ids[1::2] if len(i) > 5]
    # print(ids)
    status = [i.strip('\\n') for i in running_ids[::2][1:]]
    # print(status)
    j_running = [j for ix, j in enumerate(ids) if 'R' in status[ix] and j in j_ids]
    j_pending = [j for ix, j in enumerate(ids) if 'PD' in status[ix]]

    for jid, j_path in zip(j_ids, exp_dir_list):
        if os.path.isfile(os.path.join(j_path, 'halt.out')):
            j_finished.append(jid)

    return j_running, j_finished, j_error, j_pending


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument('--in_name', type=str, default='/checkpoint/pjaeger/bias_study_cluster/bias_study_fix_analysis_s2s3/log_test_final')
    parser.add_argument('--out_name', type=str, default='/private/home/pjaeger/fs/log/log_doublecheck_bias_study_fix_analysis_s2s3_surveillance_sheet.txt')
    parser.add_argument('--n_jobs', type=str, default=2)

    #parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
    #time.sleep(10) todo put back!
    args = parser.parse_args()
    logs_dir = args.in_name
    out_name = args.out_name
    n_total = args.n_jobs
    print('logging to: {}'.format(out_name))
    start_time = time.time()
    keep_running =True

    while keep_running:

        exp_group_dir = ('/').join(logs_dir.split('/')[:-1])
        err_files_list = [os.path.join(logs_dir, i) for i in os.listdir(logs_dir) if '.err' in i]
        out_files_list = [os.path.join(logs_dir, i) for i in os.listdir(logs_dir) if '.out' in i]
        j_ids = [i.split('/')[-1].split('_')[0] for i in out_files_list]
        j_names = {}
        for j in j_ids:
            j_names[j] = [i.split('/')[-1] for i in out_files_list if j in i][0]
        exp_dir_list = [os.path.join(exp_group_dir, ('_').join(j.split('/')[-1].split('.')[0].split('_')[1:])) for j in out_files_list]
        # n_total = len(out_files_list)


        j_running, j_finished, j_error, j_pending = check_jobs()

        j_missing = [j for j in j_ids if (j not in j_running and j not in j_finished and j not in j_error and j not in j_pending and not '.' in j)]
        n_running = len(j_running)
        n_pending = len(j_pending)
        n_finished = len(j_finished)
        n_error = len(j_error)
        n_missing = len(j_missing)
        n_total_check = n_running + n_finished + n_error + n_pending + n_missing

        overlaps = {}
        overlaps['run_finished'] =  set(j_running).intersection(j_finished)
        overlaps['run_err'] =  set(j_running).intersection(j_error)
        overlaps['err_finished'] =  set(j_error).intersection(j_finished)


        with open(out_name, "w") as f:

            f.write('start time: {} \n \n'.format(time.ctime(int(start_time))))
            f.write('last logged: {} \n \n'.format(time.ctime(int(time.time()))))

            f.write('PENDING {}/{} \n'.format(n_pending, n_total))
            f.write('RUNNING {}/{} \n'.format(n_running, n_total))
            f.write('FINISHED {}/{} \n'.format(n_finished, n_total))
            f.write('ERROR {}/{} \n'.format(n_error, n_total))
            f.write('MISSING {}/{} \n'.format(n_missing, n_total))
            f.write('TOTAL CHECK {}/{} \n \n'.format(n_total_check, n_total))

            f.write('\n \n')
            f.write('RUNNING {}/{} \n'.format(n_running, n_total))
            for r in j_running:
                f.write('{} {}\n'.format(r, j_names[r]))
            f.write('\n \n')
            f.write('FINISHED {}/{} \n'.format(n_finished, n_total))
            for r in j_finished:
                f.write('{} {}\n'.format(r, j_names[r]))
            f.write('\n \n')
            f.write('ERROR {}/{} \n'.format(n_error, n_total))
            for r in j_error:
                f.write('{} {}\n'.format(r, j_names[r]))
            f.write('\n \n')
            f.write('MISSING {}/{} \n'.format(n_missing, n_total))
            for r in j_missing:
                f.write('{} {}\n'.format(r, j_names[r]))
            f.write('\n \n')
            f.write('OVERLAP \n')
            for k, v in overlaps.items():
                f.write('{} {} {} \n'.format(k, len(v), v))


            f.write('\n \n')
            f.write('job ids under surveillance:  {} \n'.format(j_ids))

        time.sleep(30)

        if n_running == 0 and n_pending == 0:
            time.sleep(60)
            j_running, j_finished, j_error, j_pending = check_jobs()
            j_missing = [j for j in j_ids if (j not in j_running and j not in j_finished and j not in j_error and j not in j_pending and not '.' in j)]
            n_running = len(j_running)
            n_pending = len(j_pending)
            n_finished = len(j_finished)
            n_error = len(j_error)
            n_missing = len(j_missing)
            n_total_check = n_running + n_finished + n_error + n_pending + n_missing

            if n_running == 0 and n_pending == 0:
                keep_running = False
                print('end logging')
                with open(out_name, "a") as f:
                    f.write('\n \n')
                    f.write('ended logging: {} \n \n'.format(time.ctime(int(time.time()))))









