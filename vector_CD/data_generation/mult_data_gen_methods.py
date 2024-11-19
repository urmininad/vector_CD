import math
import numpy as np
import scipy
import gen_data_vecCI_ext as mod1
# import vector_CD.data_generation.gen_data_vecCI_ext as mod1
import utils


def lin_f(x): return x


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
                    internal_ER=None,
                    external_ER=None,
                    random_state=None):
  #########
  def lin_f(x):
    return x

  def f2(x):
    return (x + 5. * x ** 2 * np.exp(-x ** 2 / 20.))

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
      L_internal_density = None
      L_external_density = None

  N_array = [d_micro] * d_macro
  auto_coeffs_array = [[auto] for i in range(d_macro)]
  contemp_frac_array = [contemp_frac for i in range(math.comb(d_macro, 2))]

  min_coeff = 0.1
  coupling_coeffs = list(np.arange(min_coeff, coeff + 0.1, 0.1))
  coupling_coeffs += [-c for c in coupling_coeffs]

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

  else:  # linear gaussian

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
        return self.sigma * random_state.randn(T)

      def weibull(self, T):
        # Get zero-mean sigma variance weibull distribution
        a = 2
        mean = scipy.special.gamma(1. / a + 1)
        variance = scipy.special.gamma(2. / a + 1) - scipy.special.gamma(1. / a + 1) ** 2
        return self.sigma * (random_state.weibull(a=a, size=T) - mean) / np.sqrt(variance)

      def uniform(self, T):
        # Get zero-mean sigma variance uniform distribution
        mean = 0.5
        variance = 1. / 12.
        return self.sigma * (random_state.uniform(size=T) - mean) / np.sqrt(variance)

    noises = []
    for j in links:
      noise_type = random_state.choice(noise_types)
      sigma = noise_sigma[0] + (noise_sigma[1] - noise_sigma[0]) * random_state.rand()
      noises.append(getattr(noise_model(sigma), noise_type))

    if tau_max == 0:
      data, nonstationary = mod1.generate_nonlinear_contemp_timeseries(
        links=links, T=T, noises=noises, random_state=random_state)

    elif tau_max > 0:

      data_all_check, nonstationary = mod1.generate_nonlinear_contemp_timeseries(
        links=links, T=T + 10000, noises=noises, random_state=random_state)

      # If the model is stationary, break the loop
      if not nonstationary:
        data_all = data_all_check[:T]
        data = data_all
        break
      else:
        print("Trial %d: Not a stationary model" % ir)
        model_seed += 10000

  if nonstationary:
    raise ValueError("No stationary model found: %s" % data_gen)

  N = d_macro
  true_graph = utils.true_graph_from_links(macro_links, tau_max)

  return data, true_graph


#########################################
#  MRF_DAG DATA GENERATION FUNCTION
#########################################

def data_mrf_ts(data_gen,
                sam, d_macro,
                d_micro,
                coeff,
                auto,
                tau_max,
                contemp_frac,
                T,
                internal_er=None,
                external_er=None):
  if internal_er == None or external_er == None:
    raise ValueError("For MRF data gen, please input float values for internal and external densities")

  if d_macro == 3:
    L = 2
  else:
    L = d_macro

  # N_array = [d_micro]*d_macro
  auto_coeffs = [auto]  # for i in range(d_macro)]
  # contemp_frac_array = [contemp_frac for i in range(math.comb(d_macro,2))]

  min_coeff = 0.1
  coupling_coeffs = list(np.arange(min_coeff, coeff + 0.1, 0.1))
  coupling_coeffs += [-c for c in coupling_coeffs]

  coupling_funcs = [lin_f]

  model_seed = sam
  if tau_max == 0:
    range_stat = 1
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
    true_graph = utils.true_graph_from_links(links, tau_max)

    cov_list = utils.list_of_cov_mats(d_macro, d_micro, internal_er, random_state)

    if tau_max == 0:
      data, nonstationary = utils.structural_causal_process_MRF(links, d_micro, T, cross_prob=external_er,
                                                                wmin=0.1,
                                                                wmax=coeff,
                                                                auto=auto,
                                                                cov_list=cov_list,
                                                                noises=None,
                                                                transient_fraction=0.2,
                                                                random_state=random_state)

    elif tau_max > 0:

      data_all_check, nonstationary = utils.structural_causal_process_MRF(links, d_micro, T + 1000,
                                                                          cross_prob=external_er,
                                                                          wmin=0.1,
                                                                          wmax=coeff,
                                                                          auto=auto,
                                                                          cov_list=cov_list,
                                                                          noises=None,
                                                                          transient_fraction=0.2,
                                                                          random_state=random_state)

      # If the model is stationary, break the loop
      if not nonstationary:
        data_all = data_all_check[:T]
        data = data_all
        break
      else:
        print("Trial %d: Not a stationary model" % ir)
        model_seed += 10000

  if nonstationary:
    raise ValueError("No stationary model found: %s" % data_gen)

  return data, true_graph
