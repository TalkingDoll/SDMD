# Stochastic Dynamic Mode Decomposition (SDMD)

This repository contains the implementation of Stochastic Dynamic Mode Decomposition (SDMD), a data-driven framework for approximating the Koopman semigroup in stochastic dynamical systems, as presented in the paper:

**A Data-Driven Framework for Koopman Semigroup Estimation in Stochastic Dynamical Systems**
*Yuanchao Xu, Kaidi Shao, Isao Ishikawa, Yuka Hashimoto, Nikos Logothetis, Zhongwei Shen*

* [Full Paper on arXiv](https://arxiv.org/abs/2501.13301)

## Recent Updates

The current implementation leverages PyTorch for efficient computation, particularly on GPUs. Recent updates focus on improving performance and modularity:

1.  **SDE Coefficient Estimation (`sde_coefficients_estimator.py`)**:
    * The calculation of Stochastic Differential Equation (SDE) coefficients ($b(x), \sigma(x)$) is now modularized. `solver_sdmd_torch_gpu2.py` calls the dedicated `sde_coefficients_estimator.py` script instead of using embedded code for computing SDE's coefficients.

2.  **GPU Parallelization Enhancements + Numerical Stability Improvement (`solver_sdmd_torch_gpu3.py`)**:
    * **`compute_dPsi_X` Function**: Optimized for GPU parallelism. Nested loops over samples and features were replaced with broadcasted tensor operations, allowing the entire `dPsi_X` (related to the action of the generator on basis functions) to be computed efficiently in parallel.
    * **`get_derivatives` Function**: Jacobian and Hessian computations (required for the $\mathcal{A}\psi$ terms) now use a batched approach (`torch.func.jacrev`). Inputs are split into mini-batches, derivatives are computed once per batch, and results are concatenated, significantly speeding up the process compared to per-feature loops.
    * The **`compute_generator_L`** function (related to calculating the generator approximation matrix $A_N = G^{-1}H$ or the SDMD update $\hat{G}^{-1}\hat{H}$) now uses **Cholesky factorization** instead of the pseudoinverse ($\dagger$) or direct inversion ($\hat{G}^{-1}$). This is often preferred for better numerical stability when dealing with potentially ill-conditioned Gram matrices ($\hat{G}$).

3.  **Covergence of Training Loss Improvement (`solver_sdmd_torch_gpu4.py`)**:
    * Used `einsum` in `compute_dPsi_X`.
    * Now the value of training loss converges much faster and more stable.

*(Note: `solver_sdmd_torch_gpu4.py` is still under testing.)*


## References

If you use SDMD or this code in your research, please cite the following paper:

```bibtex
@misc{xu2025datadrivenframeworkkoopmansemigroup,
      title={A Data-Driven Framework for Koopman Semigroup Estimation in Stochastic Dynamical Systems},
      author={Yuanchao Xu and Kaidi Shao and Isao Ishikawa and Yuka Hashimoto and Nikos Logothetis and Zhongwei Shen},
      year={2025},
      eprint={2501.13301},
      archivePrefix={arXiv},
      primaryClass={math.DS},
      url={[https://arxiv.org/abs/2501.13301](https://arxiv.org/abs/2501.13301)},
}
```

