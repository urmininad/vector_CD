import numpy as np
from itertools import product
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA

from tigramite.pcmci import PCMCI
from tigramite.pcmci_base import PCMCIbase
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.data_processing as pp

import vector_CD.data_generation.gen_data_vecCI_ext as gen
from vector_CD.cond_ind_tests.parcorr_mult_regularized import ParCorrMult
from .pcmci_dep_score import PCMCI as pcmci_condsets

def lin_f(x):
	return x

# Global Arguments for linear non-time series model
#-----------------------------
coupling_funcs = [lin_f]
# auto_coeffs_array =  [[0], [0],[0]]
# tau_max = 0
# contemp_frac_array = [1.0, 1.0, 1.0]
# contemp_frac = 1.0
#-----------------------------

#############################################################
##############        HELPER FUNCS           ##############
#############################################################


def agg_data_by_method(agg_method, data, d_micro, d_macro):

    if agg_method == 'avg':
        partition = [d_micro]*d_macro
        for m,n in enumerate(partition):
            agg_m = np.mean([data[:,i] for i in range(sum(partition[:m]), sum(partition[:m+1]))],axis=0)
            if m==0:
                data_agg = agg_m
            else:
                data_agg = np.vstack((data_agg,agg_m))
            vector_vars = None
        return np.transpose(data_agg), vector_vars

    elif 'pca' in agg_method:
        p_comps  = int(agg_method.replace('pca_',''))
        T, _ = data.shape
        pca_data = np.zeros((T, int(p_comps*d_macro)))
        count=0
        for i in range(d_macro):
            X = data[:,count:count+d_micro]
            pca = PCA(n_components=p_comps).fit(X)
            pca_data[:,p_comps*i:p_comps*(i+1)] = X.dot(pca.components_.T)#.reshape((T,p_comps))
            count += d_micro
            if p_comps == 1:
                vector_vars = None
            else:
                vector_vars = vector_vars_from_Narray(d_macro,p_comps)
        return pca_data, vector_vars

def vector_vars_from_Narray(d_macro,d_micro):
    N_array = [d_micro]*d_macro

    N = len(N_array)
    if N<2:
        vector_vars = None
    else:
        vector_vars = {}
        l=0
        for i in range(N):
            j = N_array[i]
            for k in range(j):
                if k==0:
                    vector_vars[i] = [(k+l,0)] #only defining contemporaneous vector_vars here !!!
                else:
                    vector_vars[i].append((k+l,0)) #only defining contemporaneous vector_vars here !!!
            l+=j
    return vector_vars

def aggregate_data(data, partition):

    for m,n in enumerate(partition):
        agg_m = np.mean([data[:,i] for i in range(sum(partition[:m]), sum(partition[:m+1]))],axis=0)
        if m==0:
            data_agg = agg_m
        else:
            data_agg = np.vstack((data_agg,agg_m))
    return np.transpose(data_agg)

def pc_alg(data_agg, pc_alpha,vector_vars = None, ci_test = 'parcorr_gcm_gmb', verbosity = 0,cr = True):

    # PC-alg on aggregated data

    if vector_vars ==None:
        dataframe = pp.DataFrame(data_agg, analysis_mode = 'single')
        parcorr = ParCorr(significance='analytic')
        ci_test = parcorr

    else:
        dataframe  =  pp.DataFrame(data_agg, vector_vars = vector_vars, analysis_mode = 'single')

        if 'parcorr_maxcorr' in ci_test:
            ci_test = ParCorrMult(significance='analytic')
        elif ci_test == 'parcorr_gcm_gmb':
            ci_test = ParCorrMult(
                        correlation_type = 'gcm_gmb',
                        regularization_model = LinearRegression(),
                        significance = 'shuffle_test',
                        sig_blocklength=1,
                        sig_samples=200)
        else:
            raise ValueError("Multivariate CI test not included in suite")


    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ci_test,
        verbosity=verbosity)

    results = pcmci.run_pcalg(tau_max=0, pc_alpha=pc_alpha, conflict_resolution=cr)
    p_matrix = results['p_matrix']
    sepsets = results['sepsets']
    graph = results['graph']
    val_matrix = results['val_matrix']
    variable_order = np.argsort(
                                np.abs(val_matrix).sum(axis=(0,2)))[::-1]

    pcmci_base = PCMCIbase(
        dataframe=dataframe,
        cond_ind_test=ci_test,
        verbosity=verbosity)
    dag = pcmci_base._get_dag_from_cpdag(graph, variable_order)

    return sepsets,p_matrix,graph,dag


def pc_mci(data_agg, pc_alpha, tau_max, vector_vars = None, ci_test = 'parcorr_gcm_gmb', verbosity = 0,cr = True):

    # PC(MCI) on aggregated data

    if vector_vars ==None:
        dataframe = pp.DataFrame(data_agg, analysis_mode = 'single')
        parcorr = ParCorr(significance='analytic')
        ci_test = parcorr

    else:
        dataframe  =  pp.DataFrame(data_agg, vector_vars = vector_vars, analysis_mode = 'single')

        if 'parcorr_maxcorr' in ci_test:
            ci_test = ParCorrMult(significance='analytic')
        elif ci_test == 'parcorr_gcm_gmb':
            ci_test = ParCorrMult(
                        correlation_type = 'gcm_gmb',
                        regularization_model = LinearRegression(),
                        significance = 'shuffle_test',
                        sig_blocklength=1,
                        sig_samples=200)
        else:
            raise ValueError("Multivariate CI test not included in suite")


    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ci_test,
        verbosity=verbosity)

    if tau_max == 0:

        results = pcmci.run_pcalg(tau_max=0, pc_alpha=pc_alpha, conflict_resolution=cr)
        p_matrix = results['p_matrix']
        sepsets = results['sepsets']
        graph = results['graph']
        val_matrix = results['val_matrix']
        variable_order = np.argsort(
                                    np.abs(val_matrix).sum(axis=(0,2)))[::-1]
        pcmci_base = PCMCIbase(
            dataframe=dataframe,
            cond_ind_test=ci_test,
            verbosity=verbosity)
        dag = pcmci_base._get_dag_from_cpdag(graph, variable_order)

        pcmci_depscore = pcmci_condsets(
            dataframe=dataframe,
            cond_ind_test=ci_test,
            verbosity=verbosity)

        results_condsets = pcmci_depscore.condsets_for_depscore(mode='standard',pc_alpha=pc_alpha,tau_max=tau_max)
        condsets = results_condsets['condsets']

    else:

        results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)#, conflict_resolution=cr)
        p_matrix = results['p_matrix']
        sepsets = results['sepsets']
        graph = results['graph']
        val_matrix = results['val_matrix']
        variable_order = np.argsort(
                                    np.abs(val_matrix).sum(axis=(0,2)))[::-1]
        pcmci_base = PCMCIbase(
            dataframe=dataframe,
            cond_ind_test=ci_test,
            verbosity=verbosity)
        dag = pcmci_base._get_dag_from_cpdag(graph, variable_order)

        pcmci_depscore = pcmci_condsets(
            dataframe=dataframe,
            cond_ind_test=ci_test,
            verbosity=verbosity)

        results_condsets = pcmci_depscore.condsets_for_depscore(mode='contemp_conds',pc_alpha=pc_alpha,tau_max=tau_max)
        condsets = results_condsets['condsets']

        # condsets = pcmci_dep_score.condsets_for_depscore(mode='contemp_conds',pc_alpha=pc_alpha,tau_max=tau_max)

    return sepsets,p_matrix,graph,dag,condsets


def graph_to_dict(graph):
    """Helper function to convert graph to dictionary of links.

    Parameters
    ---------
    graph : array of shape (N, N, tau_max+1)
        Matrix format of graph in string format.

    Returns
    -------
    links : dict
        Dictionary of form {0:{(0, -1): o-o, ...}, 1:{...}, ...}.
    """
    N = graph.shape[0]

    links = dict([(j, []) for j in range(N)])
    #links_old = dict([(j, {}) for j in range(N)])

    for (i, j, tau) in zip(*np.where(graph!='')):
        if i<j:
            if graph[i,j,tau] == '-->':
                links[j].append(((i,-tau),)) # last comma needed to create a tuple of tuple
            elif graph[i,j,tau] == '<--':
                links[i].append(((j,-tau),))
            # else:
            #     raise ValueError('not a dag')

        #links_old[j][(i, -tau)] = graph[i,j,tau]

    return links

def links_to_binary_graph(links, tau_max,symmetrize=False):

    #set symmetrize to false if you need adjacencies to be a dict of the {child:(parent,lag)} form
    # if symmetrize is default, i.e. True, the adjacency matrix is not directional

    N = len(links)
    initial_graph = np.zeros((N, N, tau_max + 1), dtype='float')
    for j in range(N):
        for link in links[j]:
            i, tau = link[0]
            initial_graph[j, i, abs(tau)] = 1
            if symmetrize==True:
                initial_graph[i, j, abs(tau)] = 1

    return initial_graph

def shd(graph_true, graph_pred, double_for_anticausal = True):

    links_true= graph_to_dict(graph_true)
    links_pred= graph_to_dict(graph_pred)

    #print(links_true, links_pred)

    tau_max = int(graph_true.shape[2]-1)

    adj_true = links_to_binary_graph(links_true,tau_max)[:,:,0]
    adj_pred = links_to_binary_graph(links_pred,tau_max)[:,:,0]

    #print(adj_true, adj_pred)

    diff = np.abs(adj_true - adj_pred)

    #print(diff)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff)/2

def vectorize_to_fine(N_fine, N_array):

    tot_fine  = len(N_fine)
    tot_coarse = len(N_array)

    N_new = [tot_fine-tot_coarse+1]+[1]*(tot_coarse-1)
    #N_new = [tot_fine-2,1,1]
    #N_new  = N_proxy

    vector_vars = {}
    N = len(N_new)
    l=0
    for i in range(N):
        j = N_new[i]
        for k in range(j):
            if k==0:
                vector_vars[i] = [(k+l,0)] #only defining contemporaneous vector_vars here !!!
            else:
                vector_vars[i].append((k+l,0)) #only defining contemporaneous vector_vars here !!!
        l+=j

    return vector_vars


def coeff_avg_list(min_coeff,max_coeff,step_size, neg=True):
    coupling_coeffs = list(np.arange(min_coeff, max_coeff+0.1, step_size))
    coupling_range = len(coupling_coeffs)
    coeff_avg = np.zeros((coupling_range+1,))


    for j in range(coupling_range+1):
        coeff_avg[j] = (np.array(coupling_coeffs)).mean()

        if neg:
                if j < coupling_range:
                    coupling_coeffs += [-c for c in [coupling_coeffs[j]]]
        else:
            coeff_avg = np.array([coeff_avg[0]])
            break

        j+=1

    return coeff_avg


######################################################################
####### MAIN FUNCS to compute aggregation consistency scores ########
######################################################################

# Indpendence consistency score
def comp_ind_score(data,p_matrix,sepsets, N_array,pc_alpha, ci_test='parcorr_gcm_gmb'):

    #print(data.shape)
    ind_score = 0
    den=0

    if ci_test == 'parcorr_maxcorr':
        ci_test = ParCorrMult(significance='analytic')
    elif ci_test == 'parcorr_gcm_gmb':
        ci_test = ParCorrMult(
                    correlation_type = 'gcm_gmb',
                    regularization_model = LinearRegression(),
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)

    else:
        raise ValueError("ci_test not available for independence score")

    for (i1,j1) in product(range(len(N_array)),range(len(N_array))):
        if i1 > j1:
            if p_matrix[i1,j1] >= pc_alpha: #Edge deletion condition

                den+=1
                N = len(N_array)

                if N<2:
                    vector_vars = None
                else:
                    vector_vars = {}
                    l=0
                    for i in range(N):
                        j = N_array[i]
                        for k in range(j):
                            if k==0:
                                vector_vars[i] = [(k+l,0)]
                            else:
                                vector_vars[i].append((k+l,0))
                        l+=j


                dataframe_vec  =  pp.DataFrame(data, vector_vars = vector_vars, analysis_mode = 'single')
                ci_test.set_dataframe(dataframe_vec)

                pval_agg = ci_test.run_test(
                    X=[(i1,0)],
                    Y=[(j1,0)],
                    Z = sepsets[((i1,0),j1)]
                    )[1]

                if pval_agg >= pc_alpha:
                    ind_score+=1

    return ind_score,den

def gen_comp_ind_score(data,p_matrix,sepsets, d_macro, d_micro, pc_alpha, ci_test='parcorr_gcm_gmb',tau_min = 0):

    """ Generalizes ind_score to the time series setting
    """

    # N = len(N_array)

    if d_macro<2:
        vector_vars = None
    else:
        vector_vars = {}
        l=0
        for i in range(d_macro):
            # j = N_array[i]
            for k in range(d_micro):
                if k==0:
                    vector_vars[i] = [(k+l,0)]
                else:
                    vector_vars[i].append((k+l,0))
            l+=d_micro

    if ci_test == 'parcorr_maxcorr':
        ci_test = ParCorrMult(significance='analytic')
    elif ci_test == 'parcorr_gcm_gmb':
        ci_test = ParCorrMult(
                    correlation_type = 'gcm_gmb',
                    regularization_model = LinearRegression(),
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
    else:
        raise ValueError("ci_test not available for independence score")

    dataframe_vec  =  pp.DataFrame(data, vector_vars = vector_vars, analysis_mode = 'single')
    ci_test.set_dataframe(dataframe_vec)

    ind_score = 0
    den=0
    # tau_min = 0
    tau_max = int(p_matrix.shape[2]-1)
    # print("tau_max in gen_comp_ind_score", tau_max)

    for (i1,j1) in product(range(d_macro),range(d_macro)):
        for abstau in range(tau_min, tau_max + 1):

            if tau_max == 0:
                if i1>j1:
                    if p_matrix[i1,j1,abstau] >= pc_alpha: #Edge deletion condition
                        den+=1
                        pval_agg = ci_test.run_test(
                            X=[(i1,0)],
                            Y=[(j1,0)],
                            Z = sepsets[((i1,0),j1)]
                            )[1]

                        if pval_agg >= pc_alpha:
                            ind_score+=1
            else:
                if p_matrix[i1,j1,abstau] > pc_alpha: #Edge deletion condition
                    # print("~~~~ Edge deletion condition ~~~~", i1, j1, abstau)
                    # print(p_matrix)
                    den+=1
                    pval_agg = ci_test.run_test(
                        X=[(i1,-abstau)],
                        Y=[(j1,0)],
                        Z = sepsets[((i1,-abstau),j1)]
                        )[1]


                    if pval_agg >= pc_alpha:
                        ind_score+=1

    return ind_score,den


# Dependence consistency score
def gen_comp_dep_score(data, graph_agg, condsets, d_macro, d_micro, pc_alpha, ci_test='parcorr_gcm_gmb', tau_min=0):


    # _,N = data.shape

    if d_macro<2:
        vector_vars = None
    else:
        vector_vars = {}
        l=0
        for m in range(d_macro):
            for k in range(d_micro):
                if k==0:
                    vector_vars[m] = [(k+d_micro,0)]
                else:
                    vector_vars[m].append((k+l,0))
            l+=d_micro


    if ci_test == 'parcorr_maxcorr':
        ci_test = ParCorrMult(significance='analytic')
    elif ci_test == 'parcorr_gcm_gmb':
        ci_test = ParCorrMult(
                    correlation_type = 'gcm_gmb',
                    regularization_model = LinearRegression(),
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
    else:
        raise ValueError("ci_test not available for Dependence score")

    dataframe_vec  =  pp.DataFrame(data, vector_vars = vector_vars, analysis_mode = 'single')
    ci_test.set_dataframe(dataframe_vec)

    N = graph_agg.shape[0] # = d_macro
    dep_score = 0
    den = 0
    # tau_min = 0
    tau_max = int(graph_agg.shape[2]-1)

    for (i, j) in product(range(d_macro), range(d_macro)):
        for abstau in range(tau_min, tau_max + 1):

            if tau_max == 0:
                if i>j:
                    if graph_agg[i, j, abstau] != "":
                        den+=1
                        pval_agg = ci_test.run_test(
                            X=[(i,-abstau)],
                            Y=[(j,0)],
                            Z = condsets[((i,-abstau),j)]
                            )[1]

                        if pval_agg < pc_alpha:
                            dep_score+=1


            else:
                if graph_agg[i, j, abstau] != "":
                    den+=1
                    pval_agg = ci_test.run_test(
                        X=[(i,-abstau)],
                        Y=[(j,0)],
                        Z = condsets[((i,-abstau),j)]
                        )[1]

                    if pval_agg < pc_alpha:
                        dep_score+=1

    return dep_score, den


########################################################################
############################# Standard Error ###########################
########################################################################

def std_error(configs_score, boot_samples = 200):

	reps, coupling_range = configs_score.shape

	# Remove NaNs
	configs_score_no_nan = configs_score[~np.isnan(configs_score).any(axis=1), :]
	reps_update,_ = configs_score_no_nan.shape

	score_mean = np.zeros(coupling_range)
	std_error = np.zeros(coupling_range)


	for conf in range(coupling_range):
	    metric_boot = np.zeros((boot_samples,coupling_range))

	    for b in range(boot_samples):

	        # Store the unsampled values in b=0
	        rand = np.random.randint(0, reps_update, reps_update)
	        metric_boot[b][conf] = configs_score_no_nan[rand][:,conf].sum()/(reps_update)

	    score_mean[conf] = metric_boot[:,conf].mean()
	    std_error[conf] = metric_boot[:,conf].std()

	return score_mean, std_error



if __name__ == '__main__':

    pass
