import os, time, sys
from os import listdir
from os.path import isfile, join
import subprocess
import numpy as np
import math
from random import shuffle
import pickle
import socket
import ast

import tigramite
import tigramite.data_processing as pp
from sklearn.cross_decomposition import PLSRegression
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import vector_CD.data_generation.gen_data_vecCI_ext as mod1

from tigramite.independence_tests.parcorr import ParCorr
from vector_CD.cond_ind_tests.parcorr_mult_regularized import ParCorrMult
from tigramite.independence_tests.oracle_conditional_independence import OracleCI
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, MultiTaskLassoCV, ElasticNetCV, LassoLarsIC, MultiTaskElasticNetCV
from sklearn.decomposition import PCA


def vec_pcmci(data, d_macro, d_micro, cond_ind_test, tau_max, pc_alpha):

    vector_vars = vector_vars_from_Narray(d_macro,d_micro)
    dataframe = pp.DataFrame(data, vector_vars = vector_vars, analysis_mode  = 'single')
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)
    pcmcires = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha,)
    graph = pcmcires['graph']

    return graph


def avg_pcmci(data, d_macro, d_micro, cond_ind_test, tau_max, pc_alpha):

    T, _ = data.shape
    averaged_data = np.zeros((T,d_macro))
    count=0
    for i in range(d_macro):
        averaged_data[:,i] = data[:,count:count+d_micro].mean(axis=1)
        count += d_micro
    dataframe = pp.DataFrame(averaged_data)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)
    pcmcires = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha)

    graph = pcmcires['graph']

    return graph


def pca_pcmci(data, d_macro, d_micro, p_comps, cond_ind_test, tau_max, pc_alpha):

    T, _ = data.shape
    pca_data = np.zeros((T, int(p_comps*d_macro)))
    count=0
    for i in range(d_macro):
        X = data[:,count:count+d_micro]
        pca = PCA(n_components=p_comps).fit(X)
        # print("shape of data array X", X.shape)
        # print("shape of principal comp matrix", pca.components_.shape)
        # print("new var shape", pca_data[:,p_comps*i].shape )
        pca_data[:,p_comps*i:p_comps*(i+1)] = X.dot(pca.components_.T)#.reshape((T,p_comps))

        # pca_data[:,p_comps*i] = X.dot(pca.components_[:p_comps])
        count += d_micro

    if p_comps == 1:
        dataframe = pp.DataFrame(pca_data)
    else:
        N_array = [p_comps]*d_macro
        vector_vars = vector_vars_from_Narray(d_macro,p_comps)
        dataframe = pp.DataFrame(pca_data, vector_vars = vector_vars, analysis_mode = 'single')

    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)
    pcmcires = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha)

    graph = pcmcires['graph']

    return graph



def vanilla_pcmci(data, d_macro, d_micro, cond_ind_test, tau_max, pc_alpha):

    dataframe = pp.DataFrame(data)
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=cond_ind_test,
        verbosity=0)
    pcmcires = pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=tau_max,
        pc_alpha=pc_alpha)
    fine_graph = pcmcires['graph']
    N_array = [d_micro]*d_macro
    graph = mod1.coarsen_graph(fine_graph, N_array)

    return graph


#########################
# HELPERS
#########################

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

def get_cmi(reg_type, corr_type):
    if reg_type == 'ols':
        reg = LinearRegression()
    elif reg_type == 'ridge':
        reg = RidgeCV()
    elif reg_type == 'pls':
        reg = PLSRegression(n_components=2)
    else:
        raise ValueError("Unknown regression type")

    if 'max_corr' in corr_type:
        if 'shuffle' in corr_type:
            cmi = ParCorrMult(
                    correlation_type = 'max_corr',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
        else:
            cmi = ParCorrMult(
                    correlation_type = 'max_corr',
                    regularization_model = reg)

    elif 'gcm' in corr_type:
        if 'shuffle' in corr_type:
            cmi = ParCorrMult(
                    correlation_type = 'gcm',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
        elif 'gmb' in corr_type:
            cmi = ParCorrMult(
                    correlation_type = 'gcm_gmb',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
        else:
            cmi = ParCorrMult(
                    correlation_type = 'gcm',
                    regularization_model = reg)


    elif 'linear_hsic' in corr_type:
        if 'approx' in corr_type:
            cmi = ParCorrMult(
                    correlation_type = 'linear_hsic_approx',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=1000)
        elif 'kci' in corr_type:
            cmi = ParCorrMult(
                    correlation_type = 'linear_hsic_shuffle_kci',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=1000)
        else:
            cmi = ParCorrMult(
                    correlation_type = 'linear_hsic_shuffle',
                    regularization_model = reg,
                    significance = 'shuffle_test',
                    sig_blocklength=1,
                    sig_samples=200)
    return cmi

