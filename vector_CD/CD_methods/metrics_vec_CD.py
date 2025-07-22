import sys, os
import numpy as np
import pickle, pickle

# base_path = os.path.dirname(os.path.abspath(__file__))
if os.path.expanduser('~') == '/Users/urmininad':
  folder_name = os.path.expanduser('~') + '/Documents/Python/Mult_CI_Tests/Interimresults_vecCI/'
else:
  print("Check if local path to saved interim results is correct in metrics_vec_CD.py")
  folder_name = os.getcwd() + '/Interimresults_vecCI/'

def get_counts(para_setup):

    metrics = [ 'all_' + metric_type for metric_type in ['precision', 'recall']]
    metrics += [ 'adj_' + link_type + "_" + metric_type for link_type in ['anylink']
                                                       for metric_type in ['precision', 'recall']]
    # metrics +=  [ 'edgemarks_' + link_type + "_" + metric_type for link_type in ['contemp', 'anylink']
    #                                                    for metric_type in ['precision', 'recall']]
    # metrics +=  [ metric_type + "_" + link_type for link_type in ['anylink']
    #                             for metric_type in ['unoriented', 'conflicts']]

    metrics += ['computation_time']
    results = get_results(para_setup)

    if results is not None:
        orig_true_graphs = results['true_graphs']

        orig_pred_graphs = results['graphs']

        computation_time = results['computation_time']

        metrics_dict = get_numbers(metrics, orig_true_graphs, orig_pred_graphs, computation_time)
        return metrics_dict
    else:
        return None


def get_results(para_setup):

    name_string = '%s-'*len(para_setup)  # % para_setup
    name_string = name_string[:-1]
    file_name = folder_name + name_string % tuple(para_setup)

    try:
        print("load  ", file_name.replace("'", "").replace('"', '') + '.dat')
        results = pickle.load(open(file_name.replace("'", "").replace('"', '') + '.dat', 'rb'), encoding='latin1')
    except:
        print('failed '  , tuple(para_setup))
        return None

    return results


def get_masks(true_graphs):


    n_realizations, N, N, taumaxplusone = true_graphs.shape
    tau_max = taumaxplusone - 1

    contemp_cross_mask_tril = np.zeros((N,N,tau_max + 1)).astype('bool')
    contemp_cross_mask_tril[:,:,0] = np.tril(np.ones((N, N)), k=-1).astype('bool')
    any_mask = np.ones((N,N,tau_max + 1)).astype('bool')
    any_mask[:,:,0] = contemp_cross_mask_tril[:,:,0]

    # contemp_cross_mask_tril = np.repeat(contemp_cross_mask_tril.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)
    any_mask = np.repeat(any_mask.reshape(1, N,N,tau_max + 1), n_realizations, axis=0)

    return any_mask


def get_numbers(metrics, orig_true_graphs, orig_pred_graphs, computation_time, boot_samples=200):


    any_mask = get_masks(orig_true_graphs)
    n_realizations = len(orig_pred_graphs)
    metrics_dict = {}

    pred_graphs = orig_pred_graphs
    true_graphs = orig_true_graphs

    metrics_dict['adj_anylink_precision'] = (((true_graphs!="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                            ((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['adj_anylink_recall'] = (((true_graphs!="")*(pred_graphs!="")*any_mask).sum(axis=(1,2,3)),
                            ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['all_precision'] = ((((true_graphs=="-->")*(pred_graphs=="-->")+(true_graphs=="<--")*(pred_graphs=="<--"))*any_mask).sum(axis=(1,2,3)),
                            ((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['all_recall'] = ((((true_graphs=="-->")*(pred_graphs=="-->")+(true_graphs=="<--")*(pred_graphs=="<--"))*any_mask).sum(axis=(1,2,3)),
                            ((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )

    for metric in metrics_dict.keys():

        numerator, denominator = metrics_dict[metric]

        metric_boot = np.zeros(boot_samples)
        for b in range(boot_samples):
            rand = np.random.randint(0, n_realizations, n_realizations)
            metric_boot[b] = numerator[rand].sum()/denominator[rand].sum()

        metrics_dict[metric] = (numerator.sum()/denominator.sum(), metric_boot.std())

    metrics_dict['computation_time'] = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

    return metrics_dict
