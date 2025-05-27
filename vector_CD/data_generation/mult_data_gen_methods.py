from . import gen_data_vecCI_ext as mod1
from .gen_data_vecCI_ext import Graph

import numpy as np
import math
import os, time, sys
from scipy import stats
from sklearn.decomposition import PCA

from tigramite import plotting as tp
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.oracle_conditional_independence import OracleCI

from savar.model_generator import SavarGenerator
from copy import deepcopy
from savar.dim_methods import get_varimax_loadings_standard as varimax
from savar.functions import create_random_mode, check_stability
from savar.savar import SAVAR

from matplotlib import pyplot as plt

def lin_f(x): return x



#########################################
#  SAVAR DATA GENERATION FUNCTION
#########################################

def data_savar(data_gen, d_macro,d_micro, sam, T, coeff, auto, contemp_frac, tau_max):

    #########
    def lin_f(x): return x
    def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))
    #########

    auto_coeffs = list(np.arange(max(0., auto-0.3), auto+0.01, 0.05))
    min_coeff = 0.1
    coupling_coeffs = list(np.arange(min_coeff, coeff+0.1, 0.1))
    coupling_coeffs += [-c for c in coupling_coeffs]

    if 'high' in data_gen:
        L = int(1.5*d_macro)
    elif 'low' in data_gen:
        L= int(1.5*d_macro)
    else:
        L = d_macro

    if 'nonlinear' in data_gen:
        coupling_funcs = [lin_f,f2]
    else:
        coupling_funcs = [lin_f]

    # print('tau',tau_max)

    links_tigramite = mod1.generate_random_contemp_model(d_macro,
            L,
            coupling_coeffs,
            coupling_funcs,
            auto_coeffs,
            tau_max,
            contemp_frac,
            random_state=None)

    # print("links_tigramite",links_tigramite)

    links_savar = links_tigramite_to_savar(links_tigramite)

    # print('links_savar',links_savar)

    nx = int(math.sqrt(d_micro))
    ny = int(nx*d_macro)
    N = d_macro

    # nx = 30
    # ny = 90 # Each component is 30x30
    # T = 100
    # N = 3

    mode_weights = np.zeros((N, nx, ny))
    for i in range(d_macro):
        mode_weights[i, :, (i*nx):(i+1)*nx] = create_random_mode((nx, nx), random = False)

        # noise_weights[0, :, :30] = create_random_mode((30, 30), random = False)  # Random = False make modes round.
        # noise_weights[1, :, 30:60] = create_random_mode((30, 30), random = False)
        # noise_weights[2, :, 60:] = create_random_mode((30, 30), random = False)

    # modes_weights = noise_weights

    savar_model = SAVAR(links_coeffs=links_savar,
                        time_length=T,
                        mode_weights=mode_weights)
    savar_model.generate_data()
    data_field = savar_model.data_field
    data_new = reshape_data(data_field, nx, ny, N)
    data = np.concatenate(data_new).T
    true_graph = true_graph_from_links(links_tigramite,tau_max)

    return data, true_graph


#########################################
#  COARSENED_DAG DATA GENERATION FUNCTION
#########################################

def data_coarse_dag(data_gen,
                sam, d_macro,
                d_micro,
                coeff,
                auto,
                tau_max,
                contemp_frac,
                T,
                neg = None,
                internal_ER = None,
                external_ER = None,
                random_state=None):

    #########
    def lin_f(x): return x
    def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))
    #########

    if internal_ER and external_ER:
        L_internal_density = internal_ER
        L_external_density = external_ER

    else:
	    if 'highint' in data_gen:
	        L_internal_density = 'high'
	    elif 'lowint' in data_gen:
	        L_internal_density = 'low'

	    if 'highext' in data_gen:
	        L_external_density = 'high'
	    elif 'lowext' in data_gen:
	        L_external_density = 'low'

	    elif 'cbm' in data_gen:
	        L_internal_density = 'mrf_0.5'
	        L_external_density = 'full_0.3'

	    else:
	        L_internal_density= None
	        L_external_density = None

    N_array = [d_micro]*d_macro
    auto_coeffs_array = [[auto] for i in range(d_macro)]
    contemp_frac_array = [contemp_frac for i in range(math.comb(d_macro,2))]

    min_coeff = 0.1
    coupling_coeffs = list(np.arange(min_coeff, coeff+0.1, 0.1))
    coupling_range = len(coupling_coeffs)

    if neg == None or neg == 'symm':
        coupling_coeffs += [-c for c in coupling_coeffs]
    elif type(neg) == float or type(neg) == np.float64:
        for j in range(int(neg*coupling_range)):
            coupling_coeffs += [-c for c in [coupling_coeffs[j]]]
    # coupling_coeffs += [-c for c in coupling_coeffs]

    if '_linearmixed' in data_gen:

        coupling_funcs = [lin_f]

        noise_types = ['gaussian', 'weibull']
        noise_sigma = (0.5, 2)

    elif '_nonlinearmixed' in data_gen:

        coupling_funcs = [lin_f, f2]

        noise_types = ['gaussian', 'gaussian', 'weibull']
        noise_sigma = (0.5, 2)

    elif '_nonlineargaussian' in data_gen:

        coupling_funcs = [lin_f, f2]

        noise_types = ['gaussian']
        noise_sigma = (0.5, 2)

    else: # linear gaussian

        coupling_funcs = [lin_f]
        noise_types = ['gaussian']
        noise_sigma = (0.5, 2)

    # Models may be non-stationary. Hence, we iterate over a number of seeds
    # to find a stationary one regarding network topology, noises, etc
    model_seed = sam
    if tau_max == 0:
        range_stat = 1
    else:
        range_stat = 1000

    for ir in range(range_stat):

        random_state = np.random.RandomState(model_seed)

        links, macro_links = mod1.generate_random_contemp_vec_model(
            N_array,
            coupling_coeffs,
            coupling_funcs,
            auto_coeffs_array,
            tau_max,
            contemp_frac_array,
            contemp_frac,
            L_internal_density,
            L_external_density,
            random_state=None)

        class noise_model:
            def __init__(self, sigma=1):
                self.sigma = sigma
            def gaussian(self, T):
                # Get zero-mean unit variance gaussian distribution
                return self.sigma*random_state.randn(T)
            def weibull(self, T):
                # Get zero-mean sigma variance weibull distribution
                a = 2
                mean = scipy.special.gamma(1./a + 1)
                variance = scipy.special.gamma(2./a + 1) - scipy.special.gamma(1./a + 1)**2
                return self.sigma*(random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)
            def uniform(self, T):
                # Get zero-mean sigma variance uniform distribution
                mean = 0.5
                variance = 1./12.
                return self.sigma*(random_state.uniform(size=T) - mean)/np.sqrt(variance)

        noises = []
        for j in links:
            noise_type = random_state.choice(noise_types)
            sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
            noises.append(getattr(noise_model(sigma), noise_type))

        if tau_max == 0:
            data, nonstationary = mod1.generate_nonlinear_contemp_timeseries(
            links=links, T=T, noises=noises, random_state=random_state)

        elif tau_max>0:

            data_all_check, nonstationary = mod1.generate_nonlinear_contemp_timeseries(
            links=links, T=T+10000, noises=noises, random_state=random_state)

            # If the model is stationary, break the loop
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000


    if nonstationary:
        raise ValueError("No stationary model found: %s" % model)

    N = d_macro
    true_graph = true_graph_from_links(macro_links, tau_max)


    return data,true_graph



#########################################
#  MRF_DAG DATA GENERATION FUNCTION
#########################################

def data_mrf_ts(data_gen,
                sam, d_macro,
                d_micro,
                coeff,
                auto,
                neg,
                pca_weight,
                tau_max,
                contemp_frac,
                T,
                internal_er = None,
                external_er = None):

    if internal_er == None or external_er == None:
        raise ValueError("For MRF data gen, please input float values for internal and external densities")

    if d_macro==3:
        L = 2
    else:
        L = d_macro

    # N_array = [d_micro]*d_macro
    auto_coeffs = [auto] # for i in range(d_macro)]
    # contemp_frac_array = [contemp_frac for i in range(math.comb(d_macro,2))]

    min_coeff = 0.1
    coupling_coeffs = list(np.arange(min_coeff, coeff+0.1, 0.1))
    coupling_coeffs += [-c for c in coupling_coeffs]

    coupling_funcs = [lin_f]

    model_seed = sam
    if tau_max == 0:
        range_stat = 10
    else:
        range_stat = 1000

    for ir in range(range_stat):

        random_state = np.random.RandomState(model_seed)

        links = mod1.generate_random_contemp_model(
            N=d_macro,
            L=L,
            coupling_coeffs=coupling_coeffs,
            coupling_funcs=coupling_funcs,
            auto_coeffs=auto_coeffs,
            tau_max=tau_max,
            contemp_fraction=contemp_frac)
        true_graph = true_graph_from_links(links, tau_max)

        cov_list = list_of_cov_mats(d_macro, d_micro, internal_er, random_state)

        if tau_max == 0:
            data_all_check, nonstationary = structural_causal_process_MRF_PCA(links, d_micro, T, cross_prob=external_er,
                        wmin = 0.1,
                        wmax = coeff,
                        auto = auto,
                        neg = neg,
                        pca_weight = pca_weight,
                        cov_list= cov_list,
                        noises=None,
                        transient_fraction=0.2,
                        random_state = random_state)

            # If the model is stationary, break the loop
            # For nontime series too sometimes data can contain nans or inf (happens when d_micro>>1)
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000

        elif tau_max>0:

            data_all_check, nonstationary = structural_causal_process_MRF_PCA(links, d_micro, T+1000, cross_prob=external_er,
                        wmin = 0.1,
                        wmax = coeff,
                        auto = auto,
                        neg = neg,
                        pca_weight = pca_weight,
                        cov_list= cov_list,
                        noises=None,
                        transient_fraction=0.2,
                        random_state = random_state)

            # If the model is stationary, break the loop
            if not nonstationary:
                data_all = data_all_check[:T]
                data = data_all
                break
            else:
                print("Trial %d: Not a stationary model" % ir)
                model_seed += 10000

    if nonstationary:
        raise ValueError("No stationary model found: %s")

    return data, true_graph




############################################
##   SAVAR HELPERS
############################################

def links_tigramite_to_savar(links):
    for i in links.keys():
        newlist = []
        if links[i]==[]:
            pass
        else:
            for j in links[i]:
                listify = list(j)
                del listify[-1]
                newlist.append(tuple(listify))

            links[i] = newlist
    return links


def reshape_data(data_field_savar, nx, ny, N):

    '''
    takes data_field in Savar format ((nx*ny) X T dim)
    Output data_field of dimesnion (N, int(tot_grid/N), T) such that N grid_variables y_1,...y_N
    of dimension (int(tot_grid/N)) each
    can be extracted as
    y_1 = data_field[0,:, T]
    y_2 = data_field[1,:,T]
    ....

    '''

    tot_grid = int(nx*ny)

    data_field_unflatten = deepcopy(data_field_savar.reshape((int(tot_grid/N), N, -1)))
    data_field_re = deepcopy(data_field_unflatten.reshape((tot_grid,-1),order = 'F'))

    data_final = deepcopy(data_field_re.reshape((N,int(tot_grid/N),-1)))

    return data_final #data_field_re

############################################
##   GENERAL HELPERS
############################################

def true_graph_from_links(macro_links, tau_max):
    N = len(macro_links)
    true_graph = np.zeros((N, N, tau_max + 1), dtype = '<U3')
    true_graph[:] = ""
    for v in range(N):
        for parent in macro_links[v]:
            ## eg. parent = ((0, -1), .8, 'linear')
            u = parent[0][0]
            lag = parent[0][1]
            coeff = parent[1]
            # coupling = parent[2]
            if coeff != 0.:
                true_graph[u,v,abs(lag)] = "-->"
                if lag == 0:
                    true_graph[v,u,abs(lag)] = "<--"
    return true_graph

def dag_to_cpdag(dag, tau_max):

    links = dag_to_links(dag)
    T = len(links)+1
    pc_alpha = 0.01

    noises = [np.random.randn for i in links]
    test_data, _  = mod1.generate_nonlinear_contemp_timeseries(
                    links, T, noises=noises)
    dataframe = pp.DataFrame(test_data)
    cond_ind_test = OracleCI(graph = dag)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)
    pcmcires = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha)
    cpdag = pcmcires['graph']
    return cpdag

def dag_to_links(dag):
    """Helper function to convert DAG graph to dictionary of parents.

    Parameters
    ---------
    dag : array of shape (N, N, tau_max+1)
        Matrix format of graph in string format. Must be DAG.

    Returns
    -------
    parents : dict
        Dictionary of form {0:[(0, -1), ...], 1:[...], ...}.
    """
    def lin_f(x): return x
    coeff = np.random.uniform(0.1,0.5)
    N = dag.shape[0]
    parents = dict([(j, []) for j in range(N)])

    # if np.any(dag=='o-o') or np.any(dag=='x-x'):
    #     raise ValueError("graph must be DAG.")
    for (i, j, tau) in zip(*np.where(dag=='-->')):
        tau = int(tau)
        parents[j].append(((i, -tau),coeff,lin_f))

    return parents


############################################
##   HELPERS for MRF_dag data generation
############################################


def structural_causal_process_MRF_PCA(links, d_micro, T, cross_prob=0.5,
                        wmin=0.1,
                        wmax = 0.5,
                        auto = 0.1,
                        neg = 'symm',
                        pca_weight = None,
                        cov_list=None,
                        noises=None,
                        transient_fraction=0.2,
                        random_state = None):

    if random_state == None:
        random_state = np.random

    N = len(links.keys()) # N = d_macro
    transient = int(math.floor(transient_fraction*T))

    if cov_list != None:
        noises = [_generate_noise(cov_list[j], T+transient, random_state) for j in range(N)]
    elif noises is None:
        noises = [_generate_noise(np.eye(d_micro), T+transient, random_state) for j in range(N)]
    else:
        pass #ie noises have been provided by the user as a list for the macro_vars

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction*T)), (N*d_micro)):
            raise ValueError("noises.shape must match ((transient_fraction + 1)*T, d_macro*d_micro).")
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")


    ####################
    # Check parameters
    ####################
    max_lag = 0
    contemp_dag = Graph(N)

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            # if 'float' not in str(type(coeff)):
            #     raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    ####################
    ### Generate coeff matrix for cross_adjacencies
    ####################

    causal_order = contemp_dag.topologicalSort()
    coeff_mats = np.zeros((N,N, d_micro, d_micro))
    V_mats = np.zeros((N, d_micro, d_micro))  # S.v.d. of every (TXd_micro) dim macro_var X = U.D.Vt

    for j in causal_order: #range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            if var!=j:

                # For the PCA-unbiased case
                ###########################
                if pca_weight == None:
                    coeff_mats[j,var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, wmin, wmax, neg)

                # For the PCA-biased case
                ###########################
                else:
                    # print('PCA Biasing')

                    coeff_mats[j,var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, wmin, wmax, neg)
                    coeff_mats_pca = coeff_mats[j,var][1:]
                    ## pca_weight is the weighting of the 1st PC
                    newrow = pca_weight*np.array([np.random.binomial(1,cross_prob) for i in range(d_micro)])
                    ## Alternative: newrow = pca_weight*np.ones((d_micro,))


                    ##### OFFLINE PCA BIASING #####
                    # i.e.: Biasing w.r.t. PC's of the noises, not the causal variable
                    ###############################
                    if isinstance(noises, np.ndarray):
                        X = noises[:, j*d_micro:(j+1)*d_micro]
                    else:
                        X = noises[j]
                    pca_x = PCA(n_components=d_micro).fit(X)
                    V_mats[j] = pca_x.components_.T  # Second orthogonal matrix in the singular value decompostition X = U.D.Vt
                    coeff_mats[j,var] = V_mats[j] @ np.vstack([newrow,coeff_mats_pca])

                    ##### ONLINE PCA BIASING #####
                    ##############################
                    # coeff_mats[j,var] = np.vstack([newrow,coeff_mats_pca])


            else:
                neg_auto = 0. #do not symmetrize coeffs around zero
                coeff_mats[j,var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, auto, auto, neg = neg_auto)

    ####################
    ### Rewrite 'univariate' link_dict from coeff_mats for stationarity check
    ####################
    full_links = dict()
    for i in range(N*d_micro):
        full_links[i] = []
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            func = link_props[2]
            for m in range(d_micro):
                for n in range(d_micro):
                    if coeff_mats[j,var,m,n]!=0:
                        child_ind = np.arange(j*d_micro,(j+1)*d_micro)[m]
                        parent_ind = np.arange(var*d_micro, (var+1)*d_micro)[n]
                        full_links[child_ind] += [((parent_ind,lag),coeff_mats[j,var,m,n],func)]


    ####################
    ### Generate DATA
    ####################

    # causal_order = contemp_dag.topologicalSort()
    # print('causal order', causal_order)
    data = np.zeros((T+transient, int(N*d_micro)), dtype='float32')

    for j in range(N):

        if isinstance(noises, np.ndarray):
            data[:, j*d_micro:(j+1)*d_micro] = noises[:, j*d_micro:(j+1)*d_micro]
        else:
            data[:, j*d_micro:(j+1)*d_micro] = noises[j]  #[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:
            for link_props in links[j]:
                var, lag = link_props[0]
                # coeff = link_props[1] ## TODO: CHECK coefficient matrix
                func = link_props[2]
                # print('LHS shape',data[t, j*d_micro:(j+1)*d_micro].shape)
                # print('coeff_matrix shape', coeff_mats[j,var].shape)

                # Left-multiply (OLD)
                ##############
                # print("COEFF MAT",j,var, coeff_mats[j,var])
                data[t, j*d_micro:(j+1)*d_micro] += coeff_mats[j,var] @ func(data[t + lag, var*d_micro:(var+1)*d_micro])

                # Right Multiply (Convenient for weighting over PCA components)
                ############
                ### "OFFLINE PCA-biasing" ###
                # data[t, j*d_micro:(j+1)*d_micro] +=  func(data[t + lag, var*d_micro:(var+1)*d_micro]) @ coeff_mats[j,var]

                ############################
                ### "ONLINE PCA-biasing" ###
                ### TODO: But since V changes with t, this is non-stationary!!!!!
                ### TODO: Other option: Compute the stationary distribution of each variable from SCM, and use their components_ matrix
                ############################
                # if (t+lag)<d_micro:
                #     V = np.eye(d_micro)
                # else:
                #     X = data[:t + lag, var*d_micro:(var+1)*d_micro]
                #     # print('t = ', t, '; lag = ', lag, 'X shape = ',X.shape)
                #     pca_x = PCA(n_components=d_micro).fit(X)
                #     V = pca_x.components_.T
                # data[t, j*d_micro:(j+1)*d_micro] +=  func(data[t + lag, var*d_micro:(var+1)*d_micro]) @ V @ coeff_mats[j,var]

    data = data[transient:]

    # nonvalid = (np.any(np.isnan(data)) or np.any(np.isinf(data))) # TODO: stationarity check (see generate_nonlinear_contemp_timeseries in my code)
    if (mod1.check_stationarity(full_links)[0] == False or np.any(np.isnan(data)) or np.any(np.isinf(data)) or
        # np.max(np.abs(X)) > 1.e4 or
        np.any(np.abs(np.triu(np.corrcoef(data, rowvar=0), 1)) > 0.999)):

        # print("==============")
        # if mod1.check_stationarity(full_links)[0] == False:
        #     print("cond 1 for nonstationarity")
        # if np.any(np.isnan(data)):
        #     print("cond 2 for nonstationarity")
        # if np.any(np.isinf(data)):
        #     print("cond 3 for nonstationarity")
        # if np.any(np.abs(np.triu(np.corrcoef(data, rowvar=0), 1)) > 0.999): #condition to check determinism
        #     print("cond 4 for nonstationarity")
        # print("==============")

        nonstationary = True
    else:
        nonstationary = False

    return data, nonstationary#, full_links, coeff_mats





def structural_causal_process_MRF(links, d_micro, T,
                        cross_prob=0.5,
                        wmin=0.1,
                        wmax = 0.5,
                        auto = 0.1,
                        neg = 'symm',
                        cov_list=None,
                        noises=None,
                        transient_fraction=0.2,
                        random_state = None):

    if random_state == None:
        random_state = np.random

    N = len(links.keys()) # N = d_macro
    transient = int(math.floor(transient_fraction*T))

    if cov_list != None:
        noises = [_generate_noise(cov_list[j], T+transient) for j in range(N)]
    elif noises is None:
        noises = [_generate_noise(np.eye(d_micro), T+transient) for j in range(N)]
    else:
        pass #ie noises have been provided by the user as a list for the macro_vars

    if isinstance(noises, np.ndarray):
        if noises.shape != (T + int(math.floor(transient_fraction*T)), (N*d_micro)):
            raise ValueError("noises.shape must match ((transient_fraction + 1)*T, d_macro*d_micro).")
    else:
        if N != len(noises):
            raise ValueError("noises keys must match N.")


    ####################
    # Check parameters
    ####################
    max_lag = 0
    contemp_dag = Graph(N)

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            # if 'float' not in str(type(coeff)):
            #     raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)

    if contemp_dag.isCyclic() == 1:
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort()
    ####################
    ### Generate coeff matrix for cross_adjacencies
    ####################
    # NOTE: Between any pair of vars (i,j), a unique coeff matrix is drawn. So if i causes j at two different
    #  lags, the same coeff_matrix will be used. (#TODO?)

    coeff_mats = np.zeros((N,N, d_micro, d_micro))
    for j in causal_order: #range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            if var!=j:
                coeff_mats[j,var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, wmin, wmax, neg)
            else:
                neg_auto = 0. #do not symmetrize coeffs around zero
                coeff_mats[j,var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, auto, auto, neg = neg_auto)




    ####################
    ### Generate DATA
    ####################

    # causal_order = contemp_dag.topologicalSort()
    # print('causal order', causal_order)
    data = np.zeros((T+transient, int(N*d_micro)), dtype='float32')

    for j in range(N):

        if isinstance(noises, np.ndarray):
            data[:, j*d_micro:(j+1)*d_micro] = noises[:, j*d_micro:(j+1)*d_micro]
        else:
            data[:, j*d_micro:(j+1)*d_micro] = noises[j]  #[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:
            for link_props in links[j]:
                var, lag = link_props[0]
                # coeff = link_props[1] ## TODO: CHECK coefficient matrix
                func = link_props[2]
                # print('LHS shape',data[t, j*d_micro:(j+1)*d_micro].shape)
                data[t, j*d_micro:(j+1)*d_micro] += coeff_mats[j,var] @ func(data[t + lag, var*d_micro:(var+1)*d_micro])

    data = data[transient:]


    ####################
    ### Rewrite 'univariate' link_dict from coeff_mats for stationarity check
    ####################
    full_links = dict()
    for i in range(N*d_micro):
        full_links[i] = []
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            func = link_props[2]
            for m in range(d_micro):
                for n in range(d_micro):
                    if coeff_mats[j,var,m,n]!=0:
                        child_ind = np.arange(j*d_micro,(j+1)*d_micro)[m]
                        parent_ind = np.arange(var*d_micro, (var+1)*d_micro)[n]
                        full_links[child_ind] += [((parent_ind,lag),coeff_mats[j,var,m,n],func)]

    # TODO: check_stationarity only works for linear
    # TODO: Wrapper around sctrucal causal process: Pass full_links and noises (from line 382) to structural causal process?
    # rand_weight_matrix_cross as callable OR even: Just take coeff mats as arguements? For the PCA experiments>.

    # nonvalid = (np.any(np.isnan(data)) or np.any(np.isinf(data))) # TODO: stationarity check (see generate_nonlinear_contemp_timeseries in my code)
    if (mod1.check_stationarity(full_links)[0] == False or np.any(np.isnan(data)) or np.any(np.isinf(data)) or
        # np.max(np.abs(X)) > 1.e4 or
        np.any(np.abs(np.triu(np.corrcoef(data, rowvar=0), 1)) > 0.999)):
        nonstationary = True
    else:
        nonstationary = False

    return data, nonstationary#, full_links, coeff_mats



def _generate_noise(covar_matrix, T, random_state=None):
    """
    Generate a multivariate normal distribution using correlated innovations.

    Parameters
    ----------
    covar_matrix : array
        Covariance matrix of the random variables
    T : int
        Sample size
    -------
    noise : array
        Random noise generated according to covar_matrix
    """
    # random_state = np.random.RandomState(seed)
    if random_state == None:
        random_state = np.random

    # Pull out the number of nodes from the shape of the covar_matrix
    n_nodes = covar_matrix.shape[0]

    # Return the noise distribution
    return random_state.multivariate_normal(mean=np.zeros(n_nodes),
                                            cov=covar_matrix,
                                            size=T)


def rand_weight_matrix_cross(random_state, d_micro=3, connect_prob=0.5, wmin=0.1, wmax=0.5, neg = 'symm'):
    """
    :param nodes: number of nodes
    :param connect_prob: probability of an edge
    :param neg: If 'symm' then have positive and negative coeffs, if None then only positivem, if float then
    corresponds to fraction of skewness (0<neg<0.5), 0 implies no skew (only positive coeffs)
    and 1.0 implies equal frac of pos and neg coeffs

    :return: Cross variable weight matrix (need NOT be upper diagonal)
    """

    adjacency_matrix = np.zeros([d_micro, d_micro], dtype=np.int32)  # [parents, nodes]
    weight_matrix = np.zeros([d_micro, d_micro], dtype=np.float32)  # [parents, nodes]

    causal_order = np.flip(np.arange(d_micro))

    for i in range(d_micro):
        node = causal_order[i]
        potential_parents = np.arange(d_micro) #causal_order[(i + 1):]

        num_parents = random_state.binomial(n=d_micro, p=connect_prob)
        parents = random_state.choice(potential_parents, size=num_parents,
                                   replace=False)
        adjacency_matrix[parents, node] = 1

    coupling_coeffs = list(np.arange(wmin, wmax+0.1, 0.1))
    coupling_range = len(coupling_coeffs)

    if neg == None:
        pass
    elif neg == 'symm':
        coupling_coeffs += [-c for c in coupling_coeffs]
    elif type(neg) == float:
        for j in range(int(neg*coupling_range)):
            coupling_coeffs += [-c for c in [coupling_coeffs[j]]]
    else:
        raise ValueError('parameter neg not chosen')


    c_len = len(coupling_coeffs) #coupling coeffs list has been updated and length has changed


    for i in range(d_micro):
        for j in range(d_micro):
            if adjacency_matrix[i, j] == 1:
                weight_matrix[i, j] = float(coupling_coeffs[random_state.randint(0, c_len)])


    return weight_matrix

def rand_undirected_adj_matrix(rs, nodes, prob):
    """
    Args:
        rs: RandomState
        nodes: int
            Number of nodes.
        prob: liklihood of the
            presence of an edge

    Returns:
        M: np.array
            Symmetric adjacancy matrix of an undirected graph.
    """
    U = rs.uniform(low=0, high=1.0, size=(nodes, nodes))
    S = np.tril(U) + np.tril(U, -1).T

    M = np.where(S > (1-prob), 1, 0)
    np.fill_diagonal(M, 1)

    return M

def sample_mrf_prec(dim, M, rs):
    """
    Sample precision matrix of a Markov random field.
    Args:
        dim: int
            number of dimensions
        M: np.array
            array that encodes the sparsity structure of the precision matrix
            s.t. this is a valid MRF
        rs: RandomState
    Returns:
        P: np.array
            sampled precision matrix
    """

    # Sum of outer products of vectors make psd matrix
    P_list = []
    # Outer loop over rows
    for i in range(dim):
        # Inner loop over columns above diagonal
        for j in range(i+1, dim):
            if M[i, j] == 1:
                # Create sparsity mask
                m = np.zeros(dim)
                m[i] = m[j] = 1
                # Vector for outer product
                p = np.asarray([rs.random() if elem == 1 else 0.0 for elem in m])

                P_list.append(np.outer(p, p))

    P = np.sum(P_list, axis=0)
    eps = 0.01
    P += (eps * np.identity(dim))

    return P


def list_of_cov_mats(d_macro, d_micro, within_er, rs):

    cov_list = []
    for i in range(d_macro):
        # Sample internal adjacency matrix
        M = rand_undirected_adj_matrix(rs, d_micro, within_er)
        # Sample precision matrix for internal mechanism
        P = sample_mrf_prec(d_micro, M=M, rs=rs)
        E = np.linalg.inv(P)
        cov_list.append(E)

    return cov_list




