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
The causal discovery algorithms employed in the paper are the PC algorithm and the PCMCI(+) algorithm. 
`Tigramite` python package is required to run many of the experiments.
