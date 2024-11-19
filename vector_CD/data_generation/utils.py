import gen_data_vecCI_ext as mod1
from gen_data_vecCI_ext import Graph

import numpy as np
import math

############################################
##   HELPERS for coarse_dag data generation
############################################

def true_graph_from_links(macro_links, tau_max):
  N = len(macro_links)
  true_graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
  true_graph[:] = ""
  for v in range(N):
    for parent in macro_links[v]:
      ## eg. parent = ((0, -1), .8, 'linear')
      u = parent[0][0]
      lag = parent[0][1]
      coeff = parent[1]
      coupling = parent[2]
      if coeff != 0.:
        true_graph[u, v, abs(lag)] = "-->"
        if lag == 0:
          true_graph[v, u, abs(lag)] = "<--"
  return true_graph


############################################
##   HELPERS for MRF_dag data generation
############################################


def structural_causal_process_MRF(links, d_micro, T, cross_prob=0.5,
                                  wmin=0.1,
                                  wmax=0.5,
                                  auto=0.1,
                                  neg='symm',
                                  cov_list=None,
                                  noises=None,
                                  transient_fraction=0.2,
                                  random_state=None):
  if random_state == None:
    random_state = np.random

  N = len(links.keys())  # N = d_macro
  transient = int(math.floor(transient_fraction * T))

  if cov_list != None:
    noises = [_generate_noise(cov_list[j], T + transient) for j in range(N)]
  elif noises is None:
    noises = [_generate_noise(np.eye(d_micro), T + transient) for j in range(N)]
  else:
    pass  # ie noises have been provided by the user as a list for the macro_vars

  if isinstance(noises, np.ndarray):
    if noises.shape != (T + int(math.floor(transient_fraction * T)), (N * d_micro)):
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
        raise ValueError("var must be in 0..{}.".format(N - 1))
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
  # NOTE: Between any pair of vars (i,j), a unique coeff matrix is drawn. So if i causes j at two different
  #  lags, the same coeff_matrix will be used. (#TODO?)

  coeff_mats = np.zeros((N, N, d_micro, d_micro))
  for j in range(N):
    for link_props in links[j]:
      var, lag = link_props[0]
      if var != j:
        coeff_mats[j, var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, wmin, wmax, neg)
      else:
        neg_auto = 0.  # do not symmetrize coeffs around zero
        coeff_mats[j, var] = rand_weight_matrix_cross(random_state, d_micro, cross_prob, auto, auto, neg=neg_auto)

  ####################
  ### Rewrite 'univariate' link_dict from coeff_mats for stationarity check
  ####################
  full_links = dict()
  for i in range(N * d_micro):
    full_links[i] = []
  for j in range(N):
    for link_props in links[j]:
      var, lag = link_props[0]
      func = link_props[2]
      for m in range(d_micro):
        for n in range(d_micro):
          if coeff_mats[j, var, m, n] != 0:
            child_ind = np.arange(j * d_micro, (j + 1) * d_micro)[m]
            parent_ind = np.arange(var * d_micro, (var + 1) * d_micro)[n]
            full_links[child_ind] += [((parent_ind, lag), coeff_mats[j, var, m, n], func)]

  ####################
  ### Generate DATA
  ####################

  causal_order = contemp_dag.topologicalSort()
  # print('causal order', causal_order)
  data = np.zeros((T + transient, int(N * d_micro)), dtype='float32')

  for j in range(N):

    if isinstance(noises, np.ndarray):
      data[:, j * d_micro:(j + 1) * d_micro] = noises[:, j * d_micro:(j + 1) * d_micro]
    else:
      data[:, j * d_micro:(j + 1) * d_micro] = noises[j]  # [j](T+transient)

  for t in range(max_lag, T + transient):
    for j in causal_order:
      for link_props in links[j]:
        var, lag = link_props[0]
        # coeff = link_props[1] #
        func = link_props[2]
        # print('LHS shape',data[t, j*d_micro:(j+1)*d_micro].shape)
        data[t, j * d_micro:(j + 1) * d_micro] += coeff_mats[j, var] @ func(
          data[t + lag, var * d_micro:(var + 1) * d_micro])

  data = data[transient:]


  if (mod1.check_stationarity(full_links)[0] == False or np.any(np.isnan(data)) or np.any(np.isinf(data)) or
    # np.max(np.abs(X)) > 1.e4 or
    np.any(np.abs(np.triu(np.corrcoef(data, rowvar=0), 1)) > 0.999)):
    nonstationary = True
  else:
    nonstationary = False

  return data, nonstationary  # , full_links, coeff_mats


def _generate_noise(covar_matrix, T, seed=None):
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
  random_state = np.random.RandomState(seed)

  # Pull out the number of nodes from the shape of the covar_matrix
  n_nodes = covar_matrix.shape[0]

  # Return the noise distribution
  return random_state.multivariate_normal(mean=np.zeros(n_nodes),
                                          cov=covar_matrix,
                                          size=T)


def rand_weight_matrix_cross(random_state, d_micro=3, connect_prob=0.5, wmin=0.1, wmax=0.5, neg='symm'):
  """
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
    potential_parents = np.arange(d_micro)  # causal_order[(i + 1):]

    num_parents = random_state.binomial(n=d_micro, p=connect_prob)
    parents = random_state.choice(potential_parents, size=num_parents,
                                  replace=False)
    adjacency_matrix[parents, node] = 1

  coupling_coeffs = list(np.arange(wmin, wmax + 0.1, 0.1))
  coupling_range = len(coupling_coeffs)

  if neg == None:
    pass
  elif neg == 'symm':
    coupling_coeffs += [-c for c in coupling_coeffs]
  elif type(neg) == float:
    for j in range(int(neg * coupling_range)):
      coupling_coeffs += [-c for c in [coupling_coeffs[j]]]
  else:
    raise ValueError('parameter neg not chosen')

  c_len = len(coupling_coeffs)  # coupling coeffs list has been updated and length has changed

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

  M = np.where(S > (1 - prob), 1, 0)
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
    for j in range(i + 1, dim):
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
