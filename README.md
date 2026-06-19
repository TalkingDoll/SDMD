# Stochastic Dynamic Mode Decomposition (SDMD)

This repository contains the implementation of Stochastic Dynamic Mode Decomposition (SDMD), a data-driven framework for approximating the Koopman semigroup in stochastic dynamical systems, as presented in the paper:

**A Data-Driven Framework for Koopman Semigroup Estimation in Stochastic Dynamical Systems**
*Yuanchao Xu, Kaidi Shao, Isao Ishikawa, Yuka Hashimoto, Nikos Logothetis, Zhongwei Shen*

* [Full Paper on arXiv](https://arxiv.org/abs/2501.13301)

## Recent Updates

The current implementation leverages PyTorch for efficient computation, particularly on GPUs. Recent updates focus on improving performance and modularity:

1.  **SDE Coefficient Estimation (`sde_coefficients_estimator.py`)**:
    * The calculation of Stochastic Differential Equation (SDE) coefficients ($b(x), \sigma(x)$) is now modularized. `solver_sdmd_torch_gpu.py` calls the dedicated `sde_coefficients_estimator.py` script instead of using embedded code for computing SDE's coefficients.

2.  **GPU Parallelization Enhancements + Numerical Stability Improvement + Convergence of Training Loss Improvement (`solver_sdmd_torch_gpu.py`)**:
    * **`compute_dPsi_X` Function**: Optimized for GPU parallelism. Nested loops over samples and features were replaced with broadcasted tensor operations, allowing the entire `dPsi_X` (related to the action of the generator on basis functions) to be computed efficiently in parallel.
    * **`get_derivatives` Function**: Jacobian and Hessian computations (required for the $\mathcal{A}\psi$ terms) now use a batched approach (`torch.func.jacrev`). Inputs are split into mini-batches, derivatives are computed once per batch, and results are concatenated, significantly speeding up the process compared to per-feature loops.
    * The **`compute_generator_L`** function (related to calculating the generator approximation matrix $A_N = G^{-1}H$ or the SDMD update $\hat{G}^{-1}\hat{H}$) now uses **Cholesky factorization** instead of the pseudoinverse ($\dagger$) or direct inversion ($\hat{G}^{-1}$). This is often preferred for better numerical stability when dealing with potentially ill-conditioned Gram matrices ($\hat{G}$).
    * Used `einsum` in `compute_dPsi_X`. Now the value of training loss converges much faster and more stable.
    * Rewrote `get_derivatives, fit_koopman_model` to use batch processing.
    * Optimized `compute_dPsi_X` for streamlined processing
    * `solver_sdmd_torch_gpu.py` used the last trained model from last outer epoch and can also switch to use best trained model from last outer epoch.


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

