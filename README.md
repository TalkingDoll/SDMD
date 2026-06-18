# Stochastic Dynamic Mode Decomposition (SDMD)

This repository contains the implementation of Stochastic Dynamic Mode Decomposition (SDMD), a data-driven framework for approximating the Koopman semigroup in stochastic dynamical systems, as presented in the paper:

**A Data-Driven Framework for Koopman Semigroup Estimation in Stochastic Dynamical Systems**
*Yuanchao Xu, Kaidi Shao, Isao Ishikawa, Yuka Hashimoto, Nikos Logothetis, Zhongwei Shen*

* [Full Paper on arXiv](https://arxiv.org/abs/2501.13301)

## Code Overview

The current codebase is organized around PyTorch implementations of SDMD and two standard Koopman baselines. The main files and folders are:

1. **`solver_sdmd_torch_gpu.py`**:
   * Main SDMD implementation. It builds a neural dictionary, trains the Koopman model, computes the SDMD transition matrix, and provides eigenvalue/eigenfunction utilities.
   * Includes GPU-oriented batching for dictionary derivatives, generator terms, and model training.
   * Uses the modular SDE coefficient estimator when drift and diffusion terms are needed for generator-based computations.

2. **`solver_edmd_torch_gpu.py`**:
   * PyTorch EDMD baseline for learning a discrete-time Koopman operator from paired trajectory data.
   * Shares the neural dictionary structure with the SDMD code and provides utilities for Koopman matrix construction, eigendecomposition, and eigenfunction evaluation.

3. **`solver_gedmd_torch_gpu.py`**:
   * PyTorch gEDMD baseline for generator-based Koopman approximation.
   * Computes derivatives of the learned dictionary and combines them with estimated SDE drift/diffusion coefficients.

4. **`sde_coefficients_estimator.py`**:
   * Standalone neural estimator for SDE coefficients.
   * Trains an MLP to predict one-step state transitions, then estimates the drift term $b(x)$ and a diagonal diffusion approximation from the residuals.

5. **Example notebooks**:
   * `ou_process_1d_*` notebooks compare SDMD, EDMD, and gEDMD on the one-dimensional Ornstein-Uhlenbeck process using neural, Fourier, and monomial dictionaries.
   * `triple_well_2d_*` notebooks run two-dimensional triple-well experiments and method comparisons.
   * `2d_stuart_landau_sdmd_test_1.ipynb` contains a Stuart-Landau example.

6. **Experiment folders and data**:
   * `data/` stores curated trajectory data used by the examples.
   * `ou_extra/`, `triple_well_extra/`, and `stuart_landau_extra/` contain additional exploratory or archived experiment notebooks.


## References

If you use SDMD or this code in your research, please cite the following paper:

```bibtex
@article{10.1063/5.0283640,
    author = {Xu, Yuanchao and Shao, Kaidi and Ishikawa, Isao and Hashimoto, Yuka and Logothetis, Nikos and Shen, Zhongwei},
    title = {A data-driven framework for Koopman semigroup estimation in stochastic dynamical systems},
    journal = {Chaos: An Interdisciplinary Journal of Nonlinear Science},
    volume = {35},
    number = {10},
    pages = {103123},
    year = {2025},
    month = {10},
    abstract = {We present Stochastic Dynamic Mode Decomposition (SDMD), a novel data-driven framework for approximating the Koopman semigroup in stochastic dynamical systems. Unlike existing approaches, SDMD explicitly incorporates sampling time into its formulation to ensure numerical stability and precision in the presence of noise. By directly approximating the Koopman semigroup rather than its generator, SDMD avoids computationally expensive matrix exponential calculation, providing a more practically efficient pathway for analyzing stochastic dynamics. The framework also leverages neural networks for automated basis selection, minimizing manual effort while preserving computational efficiency. We establish SDMD’s theoretical foundations through rigorous convergence guarantees across three critical limits in order: large data, infinitesimal sampling time, and increasing dictionary size. Numerical experiments on canonical stochastic systems including oscillatory system, mean-reverting processes, metastable system, and a neural mass model demonstrate SDMD’s effectiveness in capturing the spectral properties of the Koopman semigroup, even in systems with complex random behavior.},
    issn = {1054-1500},
    doi = {10.1063/5.0283640},
    url = {https://doi.org/10.1063/5.0283640},
    eprint = {https://pubs.aip.org/aip/cha/article-pdf/doi/10.1063/5.0283640/20757477/103123_1_5.0283640.pdf},
}

```

