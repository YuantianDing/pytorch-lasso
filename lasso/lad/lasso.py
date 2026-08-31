from typing import Literal, Union

import torch

from .lad_lasso import _project_columns, _soft, _spectral_norm_sq


def ista_solve(
    A: torch.Tensor,
    k: int,
    lam: float = 0.05,
    max_iter: int = 100,
    tol: float = 1e-4,
    s_steps: int = 5,
    d_steps: Union[int, Literal["lstsq"]] = "lstsq",
    init: Literal["topk", "random"] = "topk",
):
    r"""Solve the (traditional) LASSO dictionary-learning problem via ISTA.

    The problem:

    .. math::
        \min_{D, S} \frac{1}{2} ||A - D S||_F^2 + \lambda ||S||_1
        \quad s.t. \quad ||d_j||_2 \le 1

    alternating ISTA (proximal gradient) steps on ``S`` with exact
    least-squares updates (or projected gradient steps) on ``D``.  Unlike
    the LAD fit of :func:`~boundlab.ops.dictlearn.lad_lasso.admm_solve`,
    the quadratic fit term penalizes large absolute residuals
    super-linearly, so the few heaviest data columns — which an l1 fit is
    free to write off as outliers — are fitted first.

    Args:
        A: Data matrix of shape (m, n); columns are data points.
        k: Number of dictionary atoms (D is (m, k), S is (k, n)).
        lam: Sparsity weight as a fraction of lambda_max = ||D^T A||_inf
            (the smallest weight that zeroes S at the initial D), so any
            value in (0, 1) is meaningful regardless of the scale of A.
        max_iter: Maximum number of outer iterations.
        tol: Stop when the relative change of the objective is below tol.
        s_steps: ISTA steps on S per outer iteration (step 1/||D||_2^2).
        d_steps: Gradient steps on D per outer iteration, or "lstsq".
        init: "topk" seeds D with the k largest data columns by l1 mass
            (normalized), so the dominant columns are exactly representable
            from iteration 0; "random" uses random data columns.

    Returns:
        D: The learned dictionary of shape (m, k), columns in the unit ball.
        S: The sparse codes of shape (k, n) with A ≈ D S.
    """
    m, n = A.shape
    device, dtype = A.device, A.dtype

    # The topk init also warm-starts S with the matching diagonal, so
    # iteration 0 reproduces the k dominant columns exactly (the truncation
    # solution) and the iterations refine from there instead of from zero.
    S = torch.zeros(k, n, device=device, dtype=dtype)
    if k <= n:
        if init == "topk":
            idx = A.abs().sum(dim=0).argsort(descending=True)[:k]
        else:
            idx = torch.randperm(n, device=device)[:k]
        D = A[:, idx].clone()
        scale = D.norm(dim=0).clamp_min(1e-8)
        if init == "topk":
            S[torch.arange(k, device=device), idx] = scale
    else:
        D = torch.randn(m, k, device=device, dtype=dtype)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)
    threshold = lam * (D.T @ A).abs().max()
    prev_obj = None

    for _ in range(max_iter):
        step = 1.0 / _spectral_norm_sq(D)
        Dt_D = D.T @ D
        Dt_A = D.T @ A
        for _ in range(s_steps):
            S = _soft(S - step * (Dt_D @ S - Dt_A), step * threshold)

        if d_steps == "lstsq":
            G = S @ S.T
            reg = 1e-6 * G.trace() / k + 1e-12
            G = G + reg * torch.eye(k, device=device, dtype=dtype)
            D = torch.linalg.solve(G, S @ A.T).T
        else:
            step_d = 1.0 / _spectral_norm_sq(S)
            for _ in range(d_steps):
                D = D - step_d * ((D @ S - A) @ S.T)
        D = _project_columns(D)

        obj = float(0.5 * (A - D @ S).square().sum() + threshold * S.abs().sum())
        if prev_obj is not None and abs(prev_obj - obj) <= tol * abs(prev_obj):
            break
        prev_obj = obj

    return D, S
