# Vector-valued Causal Discovery and Consistency-guided Aggregation
This is the code repository for the [paper](https://arxiv.org/pdf/2505.10476) "**Vector-valued Causal Discovery and Consistency-guided Aggregation**".
All the simulations studies of the paper may be found here. 
# Experiments
The experiments are categorized by topic in the **Jupyter-notebooks** folder.
1. *1_vec_CD.ipynb* illustrates and compares three approaches to vector-valued CD (Section 6 in paper).
2. *2_agg_validity.ipynb* illustrates the performance of the aggregation consistency scores (Section 7 in paper).
3. *3_adag_wrapper.ipynb* illustrates the performance of `adag` (adaptive aggregation) wrapper (Section 7 and 8 in paper).
4. *4_mult_CI_tests.ipynb* illustrates the performance of multivariate CI tests discussed in the Appendix D.1.
# Notes 
## Installation

1. Create and activate a virtual environment to run the experiments in this repository.
2. Clone the repository and install in editable mode:

```bash
git clone https://github.com/urmininad/vector_CD.git
cd vector_CD
pip install -e .
```
3. (Optional: Depending on the experiment being run): Install the SAVAR [python package](https://github.com/xtibau/savar) in the virtual environment. 

## CD algorithms
The causal discovery algorithms employed in the paper are the PC algorithm and the PCMCI(+) algorithm. 
`Tigramite` [python package](https://github.com/jakobrunge/tigramite) is required to run many of the experiments.
# References
[1] PC algorithm: Peter Spirtes, Clark Glymour, and Richard Scheines. *Causation, Prediction, and Search (1993)*.  
[2] PCMCI algorithm:  J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, D. Sejdinovic. *Detecting and quantifying causal associations in large nonlinear time series datasets. Sci. Adv. 5, eaau4996 (2019)*.  
[3] PCMCI+ algorithm: J. Runge. *Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence, UAI 2020, AUAI Press, 2020*.  
[4] SAVAR [python package](https://github.com/xtibau/savar): Spatiotemporal stochastic climate model for benchmarking causal discovery methods.
