import sys, os
import numpy as np
import pickle, pickle


folder_name = os.path.expanduser('~') + '/Documents/Python/aggregation_validity/new_interim_results/'

def get_counts(para_setup):

    metrics = [ 'all_' + metric_type for metric_type in ['precision', 'recall']]    
    metrics += [ 'adj_' + link_type + "_" + metric_type for link_type in ['anylink'] 
                                                       for metric_type in ['precision', 'recall']]
    metrics +=  [ 'edgemarks_' + link_type + "_" + metric_type for link_type in ['anylink'] 
                                                       for metric_type in ['precision', 'recall']]
    metrics +=  [ metric_type + "_" + link_type for link_type in ['anylink'] 
                                for metric_type in ['unoriented', 'conflicts']]

    metrics += ['ind_score', 'dep_score','shd','computation_time']
    results = get_results(para_setup)

    if results is not None:
        orig_true_graphs = results['true_graphs']

        orig_pred_graphs = results['graphs']

        ind_scores = results['ind_scores']

        dep_scores = results['dep_scores']

        shds = results['shds']

        computation_time = results['computation_time']

        metrics_dict = get_numbers(metrics, orig_true_graphs, orig_pred_graphs, ind_scores, dep_scores, shds, computation_time)
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


def _get_match_score(true_link, pred_link):
    if true_link == "" or pred_link == "": return 0
    count = 0
    # If left edgemark is correct add 1
    if true_link[0] == pred_link[0]:
        count += 1
    # If right edgemark is correct add 1
    if true_link[2] == pred_link[2]:
        count += 1
    return count
match_func = np.vectorize(_get_match_score, otypes=[int]) 


def _get_conflicts(pred_link):
    if pred_link == "": return 0
    count = 0
    # If left edgemark is conflict add 1
    if pred_link[0] == 'x':
        count += 1
    # If right edgemark is conflict add 1
    if pred_link[2] == 'x':
        count += 1
    return count
conflict_func = np.vectorize(_get_conflicts, otypes=[int]) 

def _get_unoriented(true_link):
    if true_link == "": return 0
    count = 0
    # If left edgemark is unoriented add 1
    if true_link[0] == 'o':
        count += 1
    # If right edgemark is unoriented add 1
    if true_link[2] == 'o':
        count += 1
    return count
unoriented_func = np.vectorize(_get_unoriented, otypes=[int]) 


def get_numbers(metrics, orig_true_graphs, orig_pred_graphs, ind_scores, dep_scores, shds, computation_time, boot_samples=200):


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
    metrics_dict['edgemarks_anylink_precision'] = ((match_func(true_graphs, pred_graphs)*any_mask).sum(axis=(1,2,3)),
                                                        2.*((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['edgemarks_anylink_recall'] = ((match_func(true_graphs, pred_graphs)*any_mask).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['unoriented_anylink'] = ((unoriented_func(true_graphs)*(any_mask)).sum(axis=(1,2,3)),
                                                        2.*((true_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    metrics_dict['conflicts_anylink'] = ((conflict_func(pred_graphs)*(any_mask)).sum(axis=(1,2,3)),
                                                            2.*((pred_graphs!="")*any_mask).sum(axis=(1,2,3)) )
    
    
    for metric in metrics_dict.keys():

        numerator, denominator = metrics_dict[metric]

        metric_boot = np.zeros(boot_samples)
        for b in range(boot_samples):
            rand = np.random.randint(0, n_realizations, n_realizations)
            metric_boot[b] = numerator[rand].sum()/denominator[rand].sum()

        metrics_dict[metric] = (numerator.sum()/denominator.sum(), metric_boot.std())



    for metric in ['ind_score', 'dep_score','shd']:
        if metric == 'ind_score':
            scores_no_nan = ind_scores[~np.isnan(ind_scores)]
            reps_update = len(scores_no_nan)

        elif metric == 'dep_score':
            scores_no_nan = dep_scores[~np.isnan(dep_scores)]
            reps_update = len(scores_no_nan)

        elif metric == 'shd':
            scores_no_nan = shds[~np.isnan(ind_scores)]
            reps_update = len(scores_no_nan)

        metric_boot = np.zeros(boot_samples)
        for b in range(boot_samples):
            rand = np.random.randint(0, reps_update, reps_update)
            metric_boot[b] = scores_no_nan[rand].sum()/(reps_update)

        metrics_dict[metric] = metric_boot.mean(), metric_boot.std()

    metrics_dict['computation_time'] = (np.mean(np.array(computation_time)), np.percentile(np.array(computation_time), [5, 95]))

    return metrics_dict
    

if __name__ == '__main__':

    pass