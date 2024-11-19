from itertools import product
import numpy as np
import math
import sys
from collections import defaultdict 
from bisect import bisect


class Graph(): 
	def __init__(self,vertices): 
		self.graph = defaultdict(list) 
		self.V = vertices 
  
	def addEdge(self,u,v): 
		self.graph[u].append(v) 
  
	def isCyclicUtil(self, v, visited, recStack): 
  
		# Mark current node as visited and  
		# adds to recursion stack 
		visited[v] = True
		recStack[v] = True
  
		# Recur for all neighbours 
		# if any neighbour is visited and in  
		# recStack then graph is cyclic 
		for neighbour in self.graph[v]: 
			if visited[neighbour] == False: 
				if self.isCyclicUtil(neighbour, visited, recStack) == True: 
					return True
			elif recStack[neighbour] == True: 
				return True
  
		# The node needs to be poped from  
		# recursion stack before function ends 
		recStack[v] = False
		return False
  
	# Returns true if graph is cyclic else false 
	def isCyclic(self): 
		visited = [False] * self.V 
		recStack = [False] * self.V 
		for node in range(self.V): 
			if visited[node] == False: 
				if self.isCyclicUtil(node,visited,recStack) == True: 
					return True
		return False
  
	# A recursive function used by topologicalSort 
	def topologicalSortUtil(self,v,visited,stack): 

	  # Mark the current node as visited. 
	  visited[v] = True

	  # Recur for all the vertices adjacent to this vertex 
	  for i in self.graph[v]: 
		  if visited[i] == False: 
			  self.topologicalSortUtil(i,visited,stack) 

	  # Push current vertex to stack which stores result 
	  stack.insert(0,v) 

	# The function to do Topological Sort. It uses recursive  
	# topologicalSortUtil() 
	def topologicalSort(self): 
		# Mark all the vertices as not visited 
		visited = [False]*self.V 
		stack =[] 

		# Call the recursive helper function to store Topological 
		# Sort starting from all vertices one by one 
		for i in range(self.V): 
		  if visited[i] == False: 
			  self.topologicalSortUtil(i,visited,stack) 

		return stack

def check_stationarity(links):
	"""Returns stationarity according to a unit root test

	Assuming a Gaussian Vector autoregressive process

	Three conditions are necessary for stationarity of the VAR(p) model:
	- Absence of mean shifts;
	- The noise vectors are identically distributed;
	- Stability condition on Phi(t-1) coupling matrix (stabmat) of VAR(1)-version  of VAR(p).
	"""


	N = len(links)
	# Check parameters
	max_lag = 0

	for j in range(N):
		for link_props in links[j]:
			var, lag = link_props[0]
			# coeff = link_props[1]
			# coupling = link_props[2]

			max_lag = max(max_lag, abs(lag))

	graph = np.zeros((N,N,max_lag))
	couplings = []

	for j in range(N):
		for link_props in links[j]:
			var, lag = link_props[0]
			coeff    = link_props[1]
			coupling = link_props[2]
			if abs(lag) > 0:
				graph[j,var,abs(lag)-1] = coeff
			couplings.append(coupling)

	stabmat = np.zeros((N*max_lag,N*max_lag))
	index = 0

	for i in range(0,N*max_lag,N):
		stabmat[:N,i:i+N] = graph[:,:,index]
		if index < max_lag-1:
			stabmat[i+N:i+2*N,i:i+N] = np.identity(N)
		index += 1

	eig = np.linalg.eig(stabmat)[0]
	# print "----> maxeig = ", np.abs(eig).max()
	if np.all(np.abs(eig) < 1.):
		stationary = True
	else:
		stationary = False

	if len(eig) == 0:
		return stationary, 0.
	else:
		return stationary, np.abs(eig).max()

def generate_nonlinear_contemp_timeseries(links, T, noises=None, random_state=None):

	if random_state is None:
		random_state = np.random

	# links must be {j:[((i, -tau), func), ...], ...}
	# coeff is coefficient
	# func is a function f(x) that becomes linear ~x in limit
	# noises is a random_state.___ function

	N = len(links.keys())
	if noises is None:
		noises = [random_state.randn for j in range(N)]

	if N != max(links.keys())+1 or N != len(noises):
		raise ValueError("links and noises keys must match N.")

	# Check parameters
	max_lag = 0
	contemp = False
	contemp_dag = Graph(N)
	causal_order = list(range(N))
	for j in range(N):
		for link_props in links[j]:
			var, lag = link_props[0]
			coeff = link_props[1]
			func = link_props[2]
			if lag == 0: contemp = True
			if var not in range(N):
				raise ValueError("var must be in 0..{}.".format(N-1))
			if 'float' not in str(type(coeff)):
				raise ValueError("coeff must be float.")
			if lag > 0 or type(lag) != int:
				raise ValueError("lag must be non-positive int.")
			max_lag = max(max_lag, abs(lag))

			# Create contemp DAG
			if var != j and lag == 0:
				contemp_dag.addEdge(var, j)
				# a, b = causal_order.index(var), causal_order.index(j)
				# causal_order[b], causal_order[a] = causal_order[a], causal_order[b]

	if contemp_dag.isCyclic() == 1: 
		raise ValueError("Contemporaneous links must not contain cycle.")

	causal_order = contemp_dag.topologicalSort() 

	transient = int(.2*T)

	X = np.zeros((T+transient, N), dtype='float32')
	for j in range(N):
		X[:, j] = noises[j](T+transient)

	for t in range(max_lag, T+transient):
		for j in causal_order:
			for link_props in links[j]:
				var, lag = link_props[0]
				# if abs(lag) > 0:
				coeff = link_props[1]
				func = link_props[2]

				X[t, j] += coeff * func(X[t + lag, var])

	X = X[transient:]

	if (check_stationarity(links)[0] == False or 
		np.any(np.isnan(X)) or 
		np.any(np.isinf(X)) or
		# np.max(np.abs(X)) > 1.e4 or
		np.any(np.abs(np.triu(np.corrcoef(X, rowvar=0), 1)) > 0.999)):
		nonstationary = True
	else:
		nonstationary = False

	return X, nonstationary

def generate_random_contemp_model(N, L, 
	coupling_coeffs, 
	coupling_funcs, 
	auto_coeffs, 
	tau_max, 
	contemp_fraction=0.,
	# num_trials=1000,
	random_state=None):

	def lin(x): return x

	if random_state is None:
		random_state = np.random

	# print links
	a_len = len(auto_coeffs)
	if type(coupling_coeffs) == float:
		coupling_coeffs = [coupling_coeffs]
	c_len  = len(coupling_coeffs)
	func_len = len(coupling_funcs)

	if tau_max == 0:
		contemp_fraction = 1.

	if contemp_fraction > 0.:
		contemp = True
		L_lagged = int((1.-contemp_fraction)*L)
		L_contemp = L - L_lagged
		if L==1: 
			# Randomly assign a lagged or contemp link
			L_lagged = random_state.randint(0,2)
			L_contemp = int(L_lagged == False)

	else:
		contemp = False
		L_lagged = L
		L_contemp = 0


	# for ir in range(num_trials):

	# Random order
	causal_order = list(random_state.permutation(N))

	links = dict([(i, []) for i in range(N)])

	# Generate auto-dependencies at lag 1
	if tau_max > 0:
		for i in causal_order:
			a = auto_coeffs[random_state.randint(0, a_len)]

			if a != 0.:
				links[i].append(((int(i), -1), float(a), lin))

	chosen_links = []
	# Create contemporaneous DAG
	contemp_links = []
	for l in range(L_contemp):

		cause = random_state.choice(causal_order[:-1])
		effect = random_state.choice(causal_order)
		while (causal_order.index(cause) >= causal_order.index(effect)
			 or (cause, effect) in chosen_links):
			cause = random_state.choice(causal_order[:-1])
			effect = random_state.choice(causal_order)
		
		contemp_links.append((cause, effect))
		chosen_links.append((cause, effect))

	# Create lagged links (can be cyclic)
	lagged_links = []
	for l in range(L_lagged):

		cause = random_state.choice(causal_order)
		effect = random_state.choice(causal_order)
		while (cause, effect) in chosen_links or cause == effect:
			cause = random_state.choice(causal_order)
			effect = random_state.choice(causal_order)
		
		lagged_links.append((cause, effect))
		chosen_links.append((cause, effect))

	# print(chosen_links)
	# print(contemp_links)
	for (i, j) in chosen_links:

		# Choose lag
		if (i, j) in contemp_links:
			tau = 0
		else:
			tau = int(random_state.randint(1, tau_max+1))
		# print tau
		# CHoose coupling
		c = float(coupling_coeffs[random_state.randint(0, c_len)])
		if c != 0:
			func = coupling_funcs[random_state.randint(0, func_len)]

			links[j].append(((int(i), -tau), c, func))

	#     # Stationarity check assuming model with linear dependencies at least for large x
	#     # if check_stationarity(links)[0]:
	#         # return links
	#     X, nonstat = generate_nonlinear_contemp_timeseries(links, 
	#         T=10000, noises=None, random_state=None)
	#     if nonstat == False:
	#         return links
	#     else:
	#         print("Trial %d: Not a stationary model" % ir)


	# print("No stationary models found in {} trials".format(num_trials))
	return links


def generate_random_contemp_vec_model(
	N_array, 
	coupling_coeffs, 
	coupling_funcs, 
	auto_coeffs_array, 
	tau_max, 
	contemp_frac_array,
	contemp_frac,
	L_internal_density = None,
	L_external_density = None,
	random_state=None):

	"""
	Returns links for a random contemporaneous VECTOR model with
	Parameters: 
	-----------
	1. N_array: array of length = (# vector variables=N); each entry corresponds to size of resp. variable. 
	2. L_internal_array: length N array for num of links within each vec var
	3. L_external_array: (N choose 2) length array for num links per vec pair
	4. coupling_coeffs: arbitrary length array of couplings coeffs to randomly choose from 
	5. coupling_funcs: arbitrary length array of couplings funcs to randomly choose from
	6. auto_coeffs_array: length N array of arbitrary length arrays of auto_corr coeffs
	7. tau_max: maximum lag length
	8. contemp_frac_array: (N choose 2) length array for fraction of contemporaneous links per vector pair
	9. contemp_frac: fraction of cont links within each vec var

	"""
	
	N = len(N_array)
	if N<=2:
		raise ValueError("More than two vector variables needed")
	if N==3:
		n = 2
	else:
		n=N
	
	#N = len(N_array)
	num_inter_var_links = math.comb(N,2)
	
	# if not L_internal_density and not L_external_density:
	# 	L_internal_array = [int(min(N_array))]*N
	# 	L_external_array = [int(min(N_array))]* (n) + list(np.zeros(math.comb(N,2)-n, dtype = int))
	# elif L_internal_density== 'high' and L_external_density== 'high':
	# 	L_internal_array = [int(1.5*min(N_array))]*N
	# 	L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	# elif L_internal_density== 'low' and L_external_density == 'low':
	# 	L_internal_array = [int(0.5*min(N_array))]*N
	# 	L_external_array = [int(0.5*(min(N_array)))]*int(n/2) + list(np.zeros(math.comb(N,2)-int(n/2), dtype = int))
	# elif not L_internal_density and L_external_density == 'high':
	# 	L_internal_array = [int(min(N_array))]*N
	# 	L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	# elif L_internal_density == 'high' and not L_external_density:
	# 	L_internal_array = [int(1.5*min(N_array))]*N
	# 	L_external_array = [int(min(N_array))]* (n) + list(np.zeros(math.comb(N,2)-n, dtype = int))
	# elif L_internal_density == 'low' and L_external_density == 'high':
	# 	L_internal_array = [int(0.5*min(N_array))]*N
	# 	L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	# elif L_internal_density== 'high' and L_external_density == 'low':
	# 	L_internal_array = [int(1.5*min(N_array))]*N
	# 	L_external_array = [int(0.5*(min(N_array)))]*int(n/2) + list(np.zeros(math.comb(N,2)-int(n/2), dtype = int))
	# else:
	# 	raise ValueError("Choose appropriate internal and external link densities")

	if not L_internal_density and not L_external_density:
	    L_internal_array = [int(min(N_array))]*N
	    L_external_array = [int(min(N_array))]* (n) + list(np.zeros(math.comb(N,2)-n, dtype = int))
    
	elif L_internal_density== 'high' and L_external_density== 'high':
	    L_internal_array = [int(1.5*min(N_array))]*N
	    L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	    
	elif L_internal_density== 'low' and L_external_density == 'low':
	    L_internal_array = [int(0.5*min(N_array))]*N
	    L_external_array = [int(0.5*(min(N_array)))]*int(n/2) + list(np.zeros(math.comb(N,2)-int(n/2), dtype = int))
	    
	elif not L_internal_density and L_external_density == 'high':
	    L_internal_array = [int(min(N_array))]*N
	    L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	    
	elif L_internal_density == 'high' and not L_external_density:
	    L_internal_array = [int(1.5*min(N_array))]*N
	    L_external_array = [int(min(N_array))]* (n) + list(np.zeros(math.comb(N,2)-n, dtype = int))

	elif not L_internal_density and L_external_density == 'low':
	    L_internal_array = [int(min(N_array))]*N
	    L_external_array = [int(0.5*(min(N_array)))]*int(n/2) + list(np.zeros(math.comb(N,2)-int(n/2), dtype = int))
	    
	elif L_internal_density == 'low' and not L_external_density:
	    L_internal_array = [int(0.5*min(N_array))]*N
	    L_external_array = [int(min(N_array))]* (n) + list(np.zeros(math.comb(N,2)-n, dtype = int))

	elif L_internal_density == 'low' and L_external_density == 'high':
	    L_internal_array = [int(0.5*min(N_array))]*N
	    L_external_array = [int(1.5*(min(N_array)))]*int(1.5*n) + list(np.zeros(math.comb(N,2)-int(1.5*n), dtype = int))
	    
	elif L_internal_density== 'high' and L_external_density == 'low':
	    L_internal_array = [int(1.5*min(N_array))]*N
	    L_external_array = [int(0.5*(min(N_array)))]*int(n/2) + list(np.zeros(math.comb(N,2)-int(n/2), dtype = int))
	    
	elif L_internal_density == 'mrf_0.5' and L_external_density == 'full_0.3':
	    
	    #####################
	    # these are default values in Simons's scbm package 
	    # i.e. internally 0.5*(total adjacencies) AND
	    # external cross edge fully connected with Erdos-Renyi prob of macro_graph = 0.3
	    #####################
	    
	    full = int(min(N_array)*(min(N_array)-1)/2)
	    L_internal_array = [int(full/2)]*N # Half the adjacencies inside a node
	    
	    full_macro = int(N*(N-1)/2) 
	    full_cross = int(min(N_array)*min(N_array))
	    L_external_array = [full_cross]* int(full_macro/3) + list(np.zeros(math.comb(N,2)-int(full_macro/3), dtype = int))

	elif type(L_internal_density) == float and type(L_external_density) == float:

		print("internal and external density are FLOATS")
		# interpret floats as Erdos-Renyi densities

		full_int = int(min(N_array)*(min(N_array)-1)/2)
		num_int_edges = int(L_internal_density*full_int)
		L_internal_array = [num_int_edges]*N

		full_cross = int(min(N_array)*min(N_array))
		num_cross_edges = int(L_external_density*full_cross)
		L_external_array = [num_cross_edges]*(n) + list(np.zeros(math.comb(N,2)-n, dtype = int))


	else:
	    raise ValueError("Choose appropriate internal and external link densities")
	    

		
	#print(L_external_array, L_internal_array)
		
	def lin(x): return x

	if random_state is None:
		random_state = np.random

	a_len = [len(auto_coeffs_array[i]) for i in range(len(auto_coeffs_array))]

	if type(coupling_coeffs) == float:
		coupling_coeffs = [coupling_coeffs]
	c_len  = len(coupling_coeffs)

	#func_len = [len(coupling_funcs[i]) for i in range(len(coupling_funcs))]
	func_len = len(coupling_funcs)

	if tau_max == 0:
		contemp_frac = 1.
		contemp_frac_array = np.zeros(num_inter_var_links)+1

	# Categorise links into contemp or lagged WITHIN a vec var    
	L_lagged_array_micro = []
	L_contemp_array_micro = []

	if contemp_frac > 0.:
		contemp_int = True

		for i in range(N):
			L_lagged_array_micro.append(int((1.-contemp_frac)*L_internal_array[i]))
			L_contemp_array_micro.append(L_internal_array[i] - L_lagged_array_micro[i])
			if L_internal_array[i]==1 and contemp_frac<1.0: 
				# Randomly assign a lagged or contemp link
				L_lagged_array_micro[i] = random_state.randint(0,2)
				L_contemp_array_micro[i] = int(L_lagged_array_micro[i] == False)

	else:
		contemp_int = False
		L_lagged_array_micro = L_internal_array
		L_contemp = [0]*N#np.zeros(N)

	# Categorise links into contemp or lagged BETWEEN vec vars
	L_lagged_array_macro = []
	L_contemp_array_macro = []

	if any(contemp_frac_array) > 0.:
		contemp_ext = True


		for i in range(num_inter_var_links):

			L_lagged_array_macro.append(int((1.-contemp_frac_array[i])*L_external_array[i]))
			L_contemp_array_macro.append(L_external_array[i] - L_lagged_array_macro[i])


			if L_external_array[i]==1 and contemp_frac_array[i]<1.0: 
				# Randomly assign a lagged or contemp link
				L_lagged_array_macro[i] = random_state.randint(0,2)
				L_contemp_array_macro[i] = int(L_lagged_array_macro[i] == False)

	else:
		contemp_ext = False
		L_lagged_array_macro = L_external_array
		L_contemp_array_macro = [0]*num_inter_var_links #np.zeros(num_inter_var_links)

	# Random causal order of vec vars 
	causal_order_macro = list(random_state.permutation(N)) #N length array


	#Random causal order of micro-vars within a vec var
	causal_order_micro = list(np.zeros(N)) #N length array
	for i in range(N):
		causal_order_micro[i] = list(random_state.permutation(N_array[i])+sum(N_array[:i])) # (N_array[i]) length array at ith pos
	# print("----")
	# print(causal_order_micro)
	# print("----")


	links = dict([(i, []) for i in range(sum(N_array))])
	macro_links = dict([(i, []) for i in range(N)])

	# Generate auto-dependencies at lag 1
	if tau_max > 0: 
		for i in range(N):

			if any(auto_coeffs_array[i]) !=0 :

				# print("non-zero aut_coeff found for N=",i)

				for j in range(N_array[i]):

					a = auto_coeffs_array[i][random_state.randint(0, a_len[i])]
					if a != 0.:
						k = sum(N_array[:i])+j #position of the jth micro-var in the ith macro-var
						links[k].append(((int(k), -1), float(a), lin))



	# Create contemporaneous DAG  and lagged links WITHIN vec var (MICRO)
	#contemp_links = []
	# lagged_links = []
	chosen_links_cont_mic = []
	chosen_links_lag_mic = []

	for k in range(N):

		if contemp_int:

			for l in range(L_contemp_array_micro[k]):


				cause = random_state.choice(causal_order_micro[k][:-1])
				effect = random_state.choice(causal_order_micro[k])

				while ((causal_order_micro[k].index(cause) >= causal_order_micro[k].index(effect))
					   or (cause, effect) in chosen_links_cont_mic):

					cause = random_state.choice(causal_order_micro[k][:-1])
					effect = random_state.choice(causal_order_micro[k])


				#contemp_links.append((cause, effect))
				chosen_links_cont_mic.append((cause, effect))

		for m in range(L_lagged_array_micro[k]):


			cause = random_state.choice(causal_order_micro[k][:-1])
			effect = random_state.choice(causal_order_micro[k])

			while (cause==effect or (cause, effect) in chosen_links_lag_mic):

				cause = random_state.choice(causal_order_micro[k][:-1])
				effect = random_state.choice(causal_order_micro[k])

			# lagged_links.append((cause, effect))
			chosen_links_lag_mic.append((cause, effect))


	 # Create contemporaneous DAG  and lagged links between vec vars (MACRO)

	chosen_links_cont_mac = []
	chosen_links_lag_mac = []

	count=0
	for i in range(N):
		for j in range(N):
			if i<j:
				if (causal_order_macro.index(i) > causal_order_macro.index(j)):
					causal_i = j
					causal_j = i

				else:
					causal_i  = i
					causal_j = j

				for l in range(L_contemp_array_macro[count]):
					cause = random_state.choice(N_array[causal_i],1)[0]+ sum(N_array[:causal_i])
					effect = random_state.choice(N_array[causal_j],1)[0]+ sum(N_array[:causal_j])

					while (cause,effect) in chosen_links_cont_mac:
						cause = random_state.choice(N_array[causal_i],1)[0]+ sum(N_array[:causal_i])
						effect = random_state.choice(N_array[causal_j],1)[0]+ sum(N_array[:causal_j])

					#contemp_links.append((cause, effect))
					chosen_links_cont_mac.append((cause, effect))

				for l in range(L_lagged_array_macro[count]):
					cause = random_state.choice(N_array[causal_i],1)[0]+ sum(N_array[:causal_i])
					effect = random_state.choice(N_array[causal_j],1)[0]+ sum(N_array[:causal_j])

					while (cause,effect) in chosen_links_lag_mac:
						cause = random_state.choice(N_array[causal_i],1)[0]+ sum(N_array[:causal_i])
						effect = random_state.choice(N_array[causal_j],1)[0]+ sum(N_array[:causal_j])

					#contemp_links.append((cause, effect))
					chosen_links_lag_mac.append((cause, effect))

				count+= 1


	lagged_links = chosen_links_lag_mic + chosen_links_lag_mac
	contemp_links = chosen_links_cont_mic + chosen_links_cont_mac


	N_fibo = [sum(N_array[:i]) for i in range(len(N_array))]
	contemp_links_macro = []
	lagged_links_macro = []

	for (i,j) in contemp_links:
		c = float(coupling_coeffs[random_state.randint(0, c_len)])
		if c != 0:
			func = coupling_funcs[random_state.randint(0, func_len)]
			tau = 0
			links[j].append(((int(i), -tau), c, func))

			pos_i = bisect(N_fibo, i)-1
			pos_j = bisect(N_fibo, j)-1

			if (pos_i,pos_j) not in contemp_links_macro and pos_i!=pos_j:
				macro_links[pos_j].append(((int(pos_i), -tau), c, func))

			contemp_links_macro.append((pos_i,pos_j))

		#if ((pos_i,pos_j) not in contemp_links_macro) and (pos_i!=pos_j):
			#macro_links[pos_j].append(((int(pos_i), -tau), c, func))

		#contemp_links_macro.append((pos_i,pos_j))


	# Running the loop separately for "chosen_links_lag_mic" and "chosen_links_lag_mac".
	# For the former set run only with tau= 1
	# Otherwise even if the auto_coeff array contains a list made up only of zeroes, 
	# in the macro_links there might still be auto-dependencies at all lags upto tau_max.
	# This happens because the lagged links within the macro_var might be at all lags upto tau_max,
	# and all these within-var links translate to autodependencies at the macro_link level.

	# CURRENT FIX: Within macro_var links taumax has been hardcoded to 1
	# such that only one type of auto-dependence appears in the macro_links
	######## CURRENT FIX: ##########################################
	################################################################ 
	for (i,j) in chosen_links_lag_mic:

		# c = float(coupling_coeffs[random_state.randint(0, c_len)])
		pos_i = bisect(N_fibo, i)-1
		pos_j = bisect(N_fibo, j)-1
		c = auto_coeffs_array[pos_i][random_state.randint(0, a_len[pos_i])]

		if c != 0:
			func = coupling_funcs[random_state.randint(0, func_len)]
			tau = 1 #int(random_state.randint(1, tau_max+1))
			links[j].append(((int(i), -tau), c, func))

			pos_i = bisect(N_fibo, i)-1
			pos_j = bisect(N_fibo, j)-1

			if (pos_i,pos_j) not in lagged_links_macro:
				macro_links[pos_j].append(((int(pos_i), -tau), c, func))

			lagged_links_macro.append((pos_i,pos_j))

	for (i,j) in chosen_links_lag_mac:
		c = float(coupling_coeffs[random_state.randint(0, c_len)])
		if c != 0:
			func = coupling_funcs[random_state.randint(0, func_len)]
			tau = int(random_state.randint(1, tau_max+1))
			links[j].append(((int(i), -tau), c, func))

			pos_i = bisect(N_fibo, i)-1
			pos_j = bisect(N_fibo, j)-1

			if (pos_i,pos_j) not in lagged_links_macro:
				macro_links[pos_j].append(((int(pos_i), -tau), c, func))

			lagged_links_macro.append((pos_i,pos_j))


	################################################################
	################################################################


	#### OLD CODE ######

	# for (i,j) in lagged_links:
	# 	c = float(coupling_coeffs[random_state.randint(0, c_len)])
	# 	if c != 0:
	# 		func = coupling_funcs[random_state.randint(0, func_len)]
	# 	tau = int(random_state.randint(1, tau_max+1))
	# 	links[j].append(((int(i), -tau), c, func))

	# 	pos_i = bisect(N_fibo, i)-1
	# 	pos_j = bisect(N_fibo, j)-1

	# 	if (pos_i,pos_j) not in lagged_links_macro:
	# 		macro_links[pos_j].append(((int(pos_i), -tau), c, func))

	# 	lagged_links_macro.append((pos_i,pos_j))


	return([links,macro_links])



def coarsen_graph(graph, N_array):

	'''
	Helper function that inputs graph and coarsens it according to N_array
	'''
	
	N = len(N_array)
	if graph.shape[0]!= sum(N_array):
		raise ValueError("Micro variables should be partitioned correctly into N_array")

	tau = graph.shape[2]
	coarse_graph = np.zeros((N,N,tau),dtype='<U3')


	for (i,j) in product(np.arange(N), np.arange(N)):
		if i<=j:
			for k in range(tau):
				subgraph = graph[sum(N_array[:i]):sum(N_array[:i+1]),sum(N_array[:j]):sum(N_array[:j+1]),k]
				count1 = np.count_nonzero(subgraph == '-->')
				count2 = np.count_nonzero(subgraph == '<--')
				#print("----",count1,count2,"-----")
				if count1>count2:
					coarse_graph[i][j][k] = '-->'
				elif count2>count1:
					coarse_graph[i][j][k] = '<--'
				else:
					if count1==0:
						coarse_graph[i][j][k] = ''
					else:
						# print("ambiguous")
						coarse_graph[i][j][k] = 'o--o'

	return(coarse_graph)

def coarsen_val_matrix(val_matrix, N_array, norm = 'fro'):
	"""
	'fro' := Frobenius Norm
	This function coarsens the val matrix but taking the frobenius norm of the submatrices
	"""
	
	N=len(N_array)
	if val_matrix.shape[0]!=sum(N_array):
		raise ValueError("Micro variables should be partitioned correctly into N_array")
	tau = val_matrix.shape[2]
	coarse_val_matrix = np.zeros((N,N,tau))
	#coarse_val_matrix = np.zeros((N,N))
	
	for (i,j) in product(np.arange(N),np.arange(N)):
		if i<=j:
			for k in range(tau):
				subgraph = val_matrix[sum(N_array[:i]):sum(N_array[:i+1]),sum(N_array[:j]):sum(N_array[:j+1]),k]
				#print(subgraph,'----\n')
				coarse_val_matrix[i][j][k]=coarse_val_matrix[j][i][k]=np.linalg.norm(subgraph, ord=norm)
				
	return(coarse_val_matrix)

def full_to_ext_graph(full_graph, ext_summ_lag):
    """
    Function that take full ts-graph output of pcmci and returns the corresposing extended summary graph
    """
    if ext_summ_lag > len(full_graph[0,0,:]):
        raise ValueError("Supplied extended summary lag must be less than or equal to tau_max of full graph")
    
    N = len(full_graph)
    print(N)
    ext_graph = np.zeros((N,N,2),dtype='<U3')
    ext_graph[:,:,0] = full_graph[:,:,0]

    for i in range(N):
        for j in range(N):
            count=0
            for k in range(1,ext_summ_lag+1):
                if full_graph[i,j,k] == '':
                    count+=0
                elif full_graph[i,j,k] == '-->':
                    count+=1
                else:
                    raise ValueError("Present vertex can only have '' or '-->' incoming edges")
            if count>0:
                ext_graph[i,j,1] = '-->'
            else:
                ext_graph[i,j,1] = ''
                
    return ext_graph  


if __name__ == '__main__':

	def lin_f(x): return x

	def nonlin_f(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

	def weibull(T): return np.random.weibull(a=2, size=T) 


############################ SCALAR MODEL ##########################

	N=10
	L=10
	links = generate_random_contemp_model(
		N=N, 
		L=L, 
		coupling_coeffs=[0.3, -0.2], 
		coupling_funcs=[lin_f, nonlin_f], 
		auto_coeffs=[0., 0.], 
		tau_max=5, 
		contemp_fraction=0.5)

#####################################################################


############################ VECTOR MODEL ###########################

	N_array = [5]*3#[3,3,3] # 3 vars of size three each  
	coupling_coeffs = [0.3]  
	coupling_funcs = [lin_f, nonlin_f]
	auto_coeffs_array =  [[0], [0],[0]]
	tau_max = 0 
	contemp_frac_array = [0.3, 0.3, 0.3]
	contemp_frac = 0.3

	links_vec = generate_random_contemp_vec_model(
	N_array, 
	coupling_coeffs, 
	coupling_funcs, 
	auto_coeffs_array, 
	tau_max, 
	contemp_frac_array,
	contemp_frac,
	L_internal_density = 0.3,#'mrf_0.5',
	L_external_density = 0.5,#'full_0.3',
	random_state=None)[0]

	N = len(links_vec.keys())
	print(links_vec)
#####################################################################


	T = 100
	data, nonstat = generate_nonlinear_contemp_timeseries(links_vec,
	 T, noises=[np.random.randn for i in range(N)])

	#new_data_1 = data[:,0:9].mean(axis=1)

	# print (np.shape(data))
	# #print(np.shape(new_data_1))
	# print(nonstat)




