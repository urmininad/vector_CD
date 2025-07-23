## This script is called by submit_agg_validity.py to compute aggregation consistency scores

import numpy as np
import sys, os, time, psutil
import pickle
import mpi
from . import funcs as ff
import vector_CD.data_generation.mult_data_gen_methods as mech
import metrics_agg as met

# --------------------------
try:
  arg = sys.argv
  num_cpus = int(arg[1])
  samples = int(arg[2])
  config_list = list(arg)[3:]
  num_configs = len(config_list)
except:
  arg = ''
  num_cpus = 10
  samples = 10
  config_list = ['avg-mrf_ts-4-5-100-0.4-0.2-.0-0.01-1-parcorr_gcm_gmb-0.3-0.5-1.0-None'] #["5-3-0.1-0.4-0.1-500-parcorr_gcm_gmb-ind-neg"]
# --------------------------

num_configs = len(config_list)
print(num_configs, "number of configs")
time_start = time.time()

def calculate(para_setup):

  para_setup_string, sam = para_setup
  paras = para_setup_string.split('-') #para_setup_string[0].split('-')
  paras = [w.replace("'","") for w in paras]

  # Parameters
  #------------
  agg_method = str(paras[0]) # 'avg' or 'pca_p' (where integer p signifies #principal comps)
  data_gen = str(paras[1])
  d_macro = int(paras[2])
  d_micro = int(paras[3])
  N_array = [d_micro]*d_macro
  T = int(paras[4])
  coeff = float(paras[5])
  auto = float(paras[6])
  contemp_frac = float(paras[7])
  pc_alpha = float(paras[8])
  tau_max = int(paras[9])
  ci_test = str(paras[10])
  internal_ER = float(paras[11]) #ER: erdos-renyi probability (internal, resp. external)
  external_ER = float(paras[12])
  neg = float(paras[13])
  if paras[14] == 'None':
    pca_weight = None
  else:
    pca_weight = float(paras[14])

  computation_time_start = time.time()


  # Generate and Aggregate Data
  #--------------------------------------
  range_nan = 5 #number of trials to find a score that isn't NaN
  for ir in range(range_nan):

      if 'coarse_dag' in data_gen:
        #pca_weight doesn't influence coarse_dag data_gen method
        data, true_graph = mech.data_coarse_dag(data_gen,sam,d_macro,d_micro,coeff,auto,
                                          tau_max,contemp_frac,T,
                                          neg,
                                          internal_ER = internal_ER,
                                          external_ER = external_ER)

      elif 'mrf_ts' in data_gen:
        data, true_graph = mech.data_mrf_ts(data_gen,sam,d_macro,d_micro,coeff,auto,
                                          neg,
                                          pca_weight,
                                          tau_max,contemp_frac,T,
                                          internal_er = internal_ER,
                                          external_er = external_ER)
      # Aggregate data
      data_agg, vector_vars = ff.agg_data_by_method(agg_method, data, d_micro, d_macro)


      # Run Time-Series CD (with nontimeseries as special case)
      #---------------------------------------

      # PC(MCI) on aggregated data
      sepsets,p_matrix,graph,dag,condsets = ff.pc_mci(data_agg, pc_alpha, tau_max,vector_vars = vector_vars,ci_test=ci_test)

      ## Aggregation consistency score computation

      ind_score,den_ind = ff.gen_comp_ind_score(data,p_matrix,sepsets, d_macro, d_micro, pc_alpha, ci_test)
      dep_score,den_dep = ff.gen_comp_dep_score(data, graph, condsets, d_macro, d_micro, pc_alpha, ci_test)
      shd_to_true_graph = ff.shd(true_graph,dag)

      if den_ind == 0 or den_dep ==0: #np.isnan(score/den):
        sam+=1000
      else:
        break

  computation_time_end = time.time()
  computation_time = computation_time_end - computation_time_start

  # print('shd', type(shd_to_true_graph), 'ind_score', type(score/den))

  return {'true_graph': true_graph,  #{'score_list':score_list,'shd_list':shd_list,
          'graph': graph,
          'shd': shd_to_true_graph,
          # 'ind_score':float(score/den),
          'ind_score':float(ind_score/den_ind),
          'dep_score':float(dep_score/den_dep),
          'computation_time': computation_time}



def process_chunks(job_id, chunk):


    results = {}
    num_here = len(chunk)
    time_start_process = time.time()
    for isam, config_sam in enumerate(chunk):
        print(config_sam)
        results[config_sam] = calculate(config_sam)

        current_runtime = (time.time() - time_start_process)/3600.
        current_runtime_hr = int(current_runtime)
        current_runtime_min = 60.*(current_runtime % 1.)
        estimated_runtime = current_runtime * num_here / (isam+1.)
        estimated_runtime_hr = int(estimated_runtime)
        estimated_runtime_min = 60.*(estimated_runtime % 1.)
        print("job_id %d index %d/%d: %dh %.1fmin / %dh %.1fmin:  %s" % (
            job_id, isam+1, num_here, current_runtime_hr, current_runtime_min,
                                    estimated_runtime_hr, estimated_runtime_min,  config_sam))
    return results


def master():

    print("Starting with num_cpus = ", num_cpus)

    all_configs = dict([(conf, {'results':{},
        "true_graphs":{},
        "graphs":{},
        "shds":{},
        "ind_scores":{},
        "dep_scores":{},
        "computation_time":{},} ) for conf in config_list])


    job_list = [(conf, i) for i in range(samples) for conf in config_list]
    num_tasks = len(job_list)
    num_jobs = min(num_cpus-1, num_tasks)


    def split(a, n):
        k, m = len(a) // n, len(a) % n
        return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


    config_chunks = split(job_list, num_jobs)
    print("num_tasks %s" % num_tasks)
    print("num_jobs %s" % num_jobs)

    ## Send
    for job_id, chunk in enumerate(config_chunks):

        print("submit %d / %d" % (job_id, len(config_chunks)))
        mpi.submit_call("process_chunks", (job_id, chunk), id = job_id)

    ## Retrieve
    for job_id, chunk in enumerate(config_chunks):
        print("\nreceive %s" % job_id)
        tmp = mpi.get_result(id=job_id)
        # tmp = {(conf, 0): array, ....}
        for conf_sam in list(tmp.keys()):
            config = conf_sam[0]
            sample = conf_sam[1]
            all_configs[config]['results'][sample] = tmp[conf_sam]
            # allresults_dists[which][job_id] = tmp[1][which]



    print("\nsaving all configs...")

    for conf in list(all_configs.keys()):

        all_configs[conf]['graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['graph'].shape, dtype='<U3')
        all_configs[conf]['true_graphs'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['true_graph'].shape, dtype='<U3')
        all_configs[conf]['shds'] = np.zeros((samples, ) + all_configs[conf]['results'][0]['shd'].shape)
        all_configs[conf]['ind_scores'] = np.zeros((samples, ) )#+ all_configs[conf]['results'][0]['ind_score'].shape)
        all_configs[conf]['dep_scores'] = np.zeros((samples, ) )#+ all_configs[conf]['results'][0]['ind_score'].shape)
        all_configs[conf]['computation_time'] = []


        for i in list(all_configs[conf]['results'].keys()):
            # all_configs[conf]['score_list'][i] = all_configs[conf]['results'][i]['score_list']
            # all_configs[conf]['shd_list'][i] = all_configs[conf]['results'][i]['shd_list']
            all_configs[conf]['graphs'][i] = all_configs[conf]['results'][i]['graph']
            all_configs[conf]['true_graphs'][i] = all_configs[conf]['results'][i]['true_graph']
            all_configs[conf]['shds'][i] = all_configs[conf]['results'][i]['shd']
            all_configs[conf]['ind_scores'][i] = all_configs[conf]['results'][i]['ind_score']
            all_configs[conf]['dep_scores'][i] = all_configs[conf]['results'][i]['dep_score']
            all_configs[conf]['computation_time'].append(all_configs[conf]['results'][i]['computation_time'])


        del all_configs[conf]['results']

        if os.path.expanduser('~') == '/home/b/b381872':
            file_name = os.path.expanduser('~') +'/work/bd1083/Interim_results/agg_validity/%s' %(conf)

        elif os.path.expanduser('~') == '/Users/urmininad':
            file_name = os.path.expanduser('~') + '/Documents/Python/aggregation_validity/new_interim_results/%s' %(conf)
            file_name_met = os.path.expanduser('~') + '/Documents/Python/aggregation_validity/metrics/%s' %(conf)

        else:
            print("New folder to save results created, change path if needed")
            newpath = os.getcwd() + '/Interimresults_agg_validity'
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            newpath_met = os.getcwd() + '/metrics_agg_validity'
            if not os.path.exists(newpath_met):
                os.makedirs(newpath_met)
            file_name = os.getcwd()+'/Interimresults_agg_validity/%s' %(conf)
            file_name_met = os.getcwd()+'/metrics_agg_validity/%s' %(conf)

        print("dump ", file_name.replace("'", "").replace('"', '') + '.dat')
        file = open(file_name.replace("'", "").replace('"', '') + '.dat', 'wb')
        pickle.dump(all_configs[conf], file, protocol=-1)
        file.close()

        # Directly compute metrics and save in much smaller dict
        para_setup_str = tuple(conf.split("-"))
        metrics = met.get_counts(para_setup_str)

        # print("metrics %s" % metrics)
        if metrics is not None:
            for metric in metrics:
                if metric != 'computation_time':
                    print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-{metrics[metric][1]: 1.2f} ")
                else:
                    # print(metrics[metric])
                    print(f"{metric:30s} {metrics[metric][0]: 1.2f} +/-[{metrics[metric][1][0]: 1.2f}, {metrics[metric][1][1]: 1.2f}]")

            print("Metrics dump ", file_name.replace("'", "").replace('"', '') + '_metrics.dat')
            file = open(file_name_met.replace("'", "").replace('"', '') + '_metrics.dat', 'wb')
            pickle.dump(metrics, file, protocol=-1)
            file.close()


    time_end = time.time()
    print('Run time in hours ', (time_end - time_start)/3600.)


mpi.run(verbose=False)



