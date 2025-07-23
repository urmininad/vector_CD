import os
from os import listdir
from os.path import isfile, join
import subprocess
from random import shuffle

submit = False # set to True and add another conditional at the end of code to run on cluster

if os.path.expanduser('~') == '/home/b/b381872':
    mypath = '/home/b/b381872/work/bd1083/Interim_results/agg_validity/'
elif os.path.expanduser('~') == '/Users/urmininad':
    mypath = '/Users/urmininad/Documents/Python/aggregation_validity/new_interim_results/'
    run_locally = True
else:
    mypath = os.getcwd() + '/Interimresults_agg_validity/'
    run_locally = True

### Change according to your setup
num_jobs = 10
run_time_hrs = 2
run_time_min = 0
num_cpus = 100
samples = 100 # Refers to number of repetitions
verbosity = 0
anyconfigurations = []
overwrite = False

# sample_config = 'avg-mrf_ts-3-5-100-0.4-0-1.0-0.01-0-parcorr_gcm_gmb-0.3-0.5-1.0-0.4'
for agg_method in ['avg']:#,'pca_1']:#['pca_1', 'pca_2','pca_3', 'pca_4']:
    for data_gen in ['mrf_ts']:#, 'coarse_dag','mrf_ts','mrf','savar']:
        for d_macro in [3]:
            for d_micro in [6]:#,4,5,6,7]:
                for T in [200]:#,1000]:
                    for coeff in [0.5]:
                        for auto in [0.3]:
                            for contemp_frac in [0.]:#[1.]:
                                for pc_alpha in [0.01]:
                                    for tau_max in [1]: #[0]:
                                        for ci_test in ['parcorr_gcm_gmb']: #'parcorr_maxcorr'
                                            for internal_ER in [0.3]:
                                                for external_ER in [0.5]:
                                                    for neg in [0., 0.2, 0.4, 0.6, 0.8, 1.]:
                                                        for pca_weight in ['None']: #[0.,0.1,0.2,0.3,0.4]: #['None']
                                                            para_setup = (agg_method,data_gen,d_macro,d_micro,T,coeff,auto,contemp_frac,
                                                                pc_alpha,tau_max,ci_test,internal_ER,external_ER,neg,pca_weight)
                                                            name = '%s-'*len(para_setup) % para_setup
                                                            name = name[:-1]
                                                            anyconfigurations += [name]



current_results_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]


already_there = []
configurations = []
for conf in anyconfigurations:
    if conf not in configurations:
        conf = conf.replace("'","")
        if (overwrite == False) and (conf + '.dat' in current_results_files):
            already_there.append(conf)
            pass
        else:
            configurations.append(conf)

for conf in configurations:
    print(conf)

num_configs = len(configurations)
print("number of todo configs ", num_configs)
print("number of existing configs ", len(already_there))
chunk_length = min(num_jobs, num_configs)   # num_configs/num_jobs
print("num_jobs %s" % num_jobs)
print("chunk_length %s" % chunk_length)
print("cpus %s" % num_cpus)
print("runtime %02.d:%02.d:00" % (run_time_hrs, run_time_min))
print("Shuffle configs to create equal computation time chunks ")
shuffle(configurations)
if num_configs == 0:
    raise ValueError("No configs to do...")

def split(a, n):
    k, m = len(a) // n, len(a) % n
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


for config_chunk in split(configurations, chunk_length):

    config_chunk = [con for con in config_chunk if con != None]
    print(config_chunk)
    config_string = str(config_chunk)[1:-1].replace(',', '').replace('"', '')
    print("-----------")
    print(config_string)

    job_list = [(conf, i) for i in range(samples) for conf in config_chunk]
    num_tasks = len(config_chunk)*samples
    num_jobs = min(num_cpus-1, num_tasks)

    print(max([len(chunk) for chunk in split(job_list, num_jobs)]))

    use_script = 'compute_agg_validity.py'

    if submit == False:
        submit_string = ["python", "compute_agg_validity.py", str(num_cpus), str(samples)] + config_chunk
        if run_locally:
            print("Run locally")
            process = subprocess.Popen(submit_string)  #,
            output = process.communicate()

    elif submit and os.path.expanduser('~') == '/home/b/b381872': #DKRZ cluster
        submit_string = ['sbatch', '--ntasks', str(num_cpus), '--time', '%02.d:%02.d:00' % (run_time_hrs, run_time_min), 'dkrz_cluster_submit.sh', use_script + " %d %d %s" %(num_cpus, samples, config_string)]  # +  config_chunk
        print("Run on cluster")
        process = subprocess.Popen(submit_string)  #,
        output = process.communicate()
    else:
        print("Not submitted.")
