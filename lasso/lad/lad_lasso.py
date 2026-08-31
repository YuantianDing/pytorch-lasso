from typing import Literal, Optional, Union

import torch


def _soft(x: torch.Tensor, t) -> torch.Tensor:
    """Elementwise soft-thresholding: sign(x) * max(|x| - t, 0)."""
    return torch.sign(x) * torch.relu(x.abs() - t)


def _spectral_norm_sq(M: torch.Tensor, n_iter: int = 10) -> torch.Tensor:
    """Estimate the squared spectral norm ||M||_2^2 by power iteration.

    Deterministic start (column energies of M) so eager and compiled runs
    follow identical trajectories; `randn` sequences differ under
    ``torch.compile``."""
    v = (M * M).sum(dim=0)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(n_iter):
        v = M.T @ (M @ v)
        v = v / v.norm().clamp_min(1e-12)
    return (M @ v).square().sum().clamp_min(1e-12)


def _project_columns(D: torch.Tensor) -> torch.Tensor:
    """Project each column of D onto the unit l2 ball."""
    return D / D.norm(dim=0, keepdim=True).clamp_min(1.0)


def _admm_step(
    A: torch.Tensor,
    D: torch.Tensor,
    S: torch.Tensor,
    U: torch.Tensor,
    E0: torch.Tensor,
    rho: float,
    lam: float,
    alpha: float,
    beta: float,
    s_steps: int,
    d_steps: Union[int, Literal["lstsq"]],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One outer ADMM iteration; E0 carries A - D @ S between iterations."""
    E = _soft(E0 + U, 1.0 / rho)
    B = A - E + U

    step_s = alpha / _spectral_norm_sq(D)
    Dt_D = D.T @ D
    Dt_B = D.T @ B
    for _ in range(s_steps):
        S = _soft(S - step_s * (Dt_D @ S - Dt_B), step_s * lam / rho)

    if d_steps == "lstsq":
        G = S @ S.T
        reg = 1e-6 * G.trace() / k + 1e-12
        G = G + reg * torch.eye(k, device=A.device, dtype=A.dtype)
        D = torch.linalg.solve(G, S @ B.T).T
    else:
        step_d = beta / _spectral_norm_sq(S)
        for _ in range(d_steps):
            D = D - step_d * ((D @ S - B) @ S.T)
    D = _project_columns(D)

    E0 = A - D @ S
    U = U + E0 - E
    return D, S, U, E0


_compiled_admm_step = None


def _admm_step_compiled():
    """Compile the ADMM step once, lazily, and reuse it across calls."""
    global _compiled_admm_step
    if _compiled_admm_step is None:
        _compiled_admm_step = torch.compile(_admm_step)
    return _compiled_admm_step


def admm_solve(
    A: torch.Tensor,
    k: int,
    max_iter: int = 1000,
    tol: float = 1e-3,
    lam: float = 0.2,
    rho: Optional[float] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    s_steps: int = 5,
    d_steps: Union[int, Literal["lstsq"]] = "lstsq",
    compiled: bool = False,
    init: Literal["random", "topk"] = "random",
):
    """Solve the LAD-LASSO problem using the ADMM algorithm.

    Given a matrix A and a regularization parameter lam, this function solves the following dictionary learning problem:

        $$min_{D, S} ||A - D S||_1 + lam * ||S||_1  s.t.  ||d_j||_2 <= 1$$

    using the Alternating Direction Method of Multipliers (ADMM) algorithm,
    with the splitting E = A - D S and scaled dual variable U.

    The algorithm computes:

    .. math::

        \\boxed{
        \\begin{aligned}
        E
        &\\leftarrow
        \\operatorname{soft}
        (A-DS+U,1/\\rho),
        \\\\[3pt]
        B
        &\\leftarrow A-E+U,
        \\\\[3pt]
        S
        &\\leftarrow
        \\operatorname{soft}
        \\left(
        S-\\alpha D^T(DS-B),
        \\alpha\\lambda/\\rho
        \\right),
        \\\\[3pt]
        D
        &\\leftarrow
        \\Pi_{\\mathcal C}
        \\left[
        D-\\beta(DS-B)S^T
        \\right],
        \\\\[3pt]
        U
        &\\leftarrow
        U+A-DS-E.
        \\end{aligned}
        }

    The S and D updates are repeated `s_steps` and `d_steps` times per outer
    iteration. If `d_steps` is set to "lstsq", the dictionary update step is
    solved exactly via regularized normal equations instead of gradient steps.

    Note on step sizes: `alpha` and `beta` are *normalized* steps. The actual
    gradient steps used are alpha / ||D||_2^2 and beta / ||S||_2^2 (spectral
    norms estimated by power iteration each outer iteration), so any value in
    (0, 1] is guaranteed-stable regardless of the size or conditioning of A.

    Args:
        A: Data matrix of shape (m, n); columns are data points.
        k: Number of dictionary atoms (D is (m, k), S is (k, n)).
        max_iter: Maximum number of outer ADMM iterations.
        tol: Stop when the primal residual ||A - DS - E||_1 / ||A||_1 < tol
            (entrywise l1 norms, matching the LAD fit term).
        lam: Sparsity weight on S (relative to the l1 fit term).
        rho: ADMM penalty parameter; the residual threshold in the E update
            is 1/rho, which must sit well below the typical entry magnitude
            of A. None (default) uses rho = 10 / mean(|A|), which places the
            threshold at ~10% of the mean absolute entry.
        alpha: Normalized step size for the S (sparse code) update.
        beta: Normalized step size for the D update when `d_steps` is an int.
        s_steps: Prox-gradient (ISTA) steps on S per outer iteration.
        d_steps: Gradient steps on D per outer iteration, or "lstsq".
        compiled: Run each outer iteration through ``torch.compile`` (one
            shared compilation, triggered lazily on first use). Worthwhile
            for large A over many iterations; leave False for small or
            one-off problems where compile latency dominates.
        init: "topk" seeds D with the k largest data columns by l1 mass
            (normalized) and warm-starts S with the matching diagonal, so
            iteration 0 is exactly the top-k truncation solution; "random"
            (default) uses random data columns with S = 0.

    For large problems (e.g. A of shape (1000, 40000)), k = 2 * m, the
    default lam / rho / alpha / s_steps, d_steps="lstsq", and max_iter of
    200-500 work well; each outer iteration costs ~15 matmuls of size
    m x k x n (~4 s on a 20-core CPU at that size, fp32).

    Returns:
        D: The learned dictionary of shape (m, k), columns in the unit ball.
        S: The sparse codes of shape (k, n) with A ≈ D S.
    """
    m, n = A.shape
    device, dtype = A.device, A.dtype
    if rho is None:
        rho = 10.0 / float(A.abs().mean().clamp_min(1e-12))

    # Initialize D from data columns (standard for dictionary learning):
    # random ones, or the k heaviest by l1 mass with init="topk".  The topk
    # init also warm-starts S with the matching diagonal, so iteration 0
    # already reproduces the k dominant columns exactly (the truncation
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
    U = torch.zeros_like(A)
    A_norm = A.abs().sum().clamp_min(1e-12)

    step = _admm_step_compiled() if compiled else _admm_step
    E0 = A - D @ S
    for _ in range(max_iter):
        D, S, U, E0 = step(
            A, D, S, U, E0, rho, lam, alpha, beta, s_steps, d_steps, k
        )
        if E0.abs().sum() / A_norm < tol:
            break

    return D, S

def _l1_grad(R: torch.Tensor, kind: str, delta: float) -> torch.Tensor:
    """Gradient (or subgradient) of the l1 approximation, elementwise."""
    if kind == "huber":
        return torch.clamp(R / delta, -1.0, 1.0)
    if kind == "smooth_l1":
        return R / torch.sqrt(R * R + delta * delta)
    return torch.sign(R)


def ista_solve(
    A: torch.Tensor,
    k: int,
    lam: float = 0.2,
    max_iter: int = 1000,
    tol: float = 1e-4,
    l1_approx: Literal["huber", "smooth_l1", "l1"] = "huber",
    alpha: float = 1.0,
    delta: Optional[float] = None,
    s_steps: int = 5,
    d_steps: int = 1,
):
    """Solve the LAD-LASSO problem by proximal gradient descent (ISTA).

    Minimizes the same objective as :func:`admm_solve`,

        $$min_{D, S} ||A - D S||_1 + lam * ||S||_1  s.t.  ||d_j||_2 <= 1$$

    but instead of splitting off the l1 fit term, replaces it with a smooth
    approximation h_delta and alternates prox-gradient steps on S and
    projected gradient steps on D (PALM-style):

        S <- soft(S + step_s * D^T h'(A - D S), step_s * lam)
        D <- Pi_C[D + step_d * h'(A - D S) S^T]

    `l1_approx` selects h: "huber" is the piecewise-quadratic Huber function
    (curvature 1/delta near 0), "smooth_l1" is the pseudo-Huber
    sqrt(x^2 + delta^2) - delta, and "l1" uses the exact sign subgradient
    (fixed-step subgradient method: converges only to an O(step)
    neighborhood; provided as a reference).

    As in :func:`admm_solve`, `alpha` is a normalized step: the actual steps
    are alpha * delta / ||D||_2^2 and alpha * delta / ||S||_2^2, i.e.
    1/(Lipschitz constant of the smoothed gradient) when alpha = 1.

    Args:
        A: Data matrix of shape (m, n); columns are data points.
        k: Number of dictionary atoms (D is (m, k), S is (k, n)).
        lam: Sparsity weight on S (same scale as in :func:`admm_solve`).
        max_iter: Maximum number of outer iterations.
        tol: Stop when the relative change of the objective is below tol.
        l1_approx: Smooth approximation of the l1 fit term (see above).
        alpha: Normalized step size for both S and D updates.
        delta: Smoothing width of h; entries with residual below ~delta are
            treated quadratically. None (default) uses 0.1 * mean(|A|),
            matching the 1/rho threshold that admm_solve's default implies.
        s_steps: Prox-gradient steps on S per outer iteration.
        d_steps: Projected gradient steps on D per outer iteration.

    Returns:
        D: The learned dictionary of shape (m, k), columns in the unit ball.
        S: The sparse codes of shape (k, n) with A ≈ D S.
    """
    m, n = A.shape
    device, dtype = A.device, A.dtype
    if delta is None:
        delta = 0.1 * float(A.abs().mean().clamp_min(1e-12))

    if k <= n:
        idx = torch.randperm(n, device=device)[:k]
        D = A[:, idx].clone()
    else:
        D = torch.randn(m, k, device=device, dtype=dtype)
    D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-8)

    S = torch.zeros(k, n, device=device, dtype=dtype)
    prev_obj = None

    for _ in range(max_iter):
        step_s = alpha * delta / _spectral_norm_sq(D)
        for _ in range(s_steps):
            G = _l1_grad(A - D @ S, l1_approx, delta)
            S = _soft(S + step_s * (D.T @ G), step_s * lam)

        step_d = alpha * delta / _spectral_norm_sq(S)
        for _ in range(d_steps):
            G = _l1_grad(A - D @ S, l1_approx, delta)
            D = _project_columns(D + step_d * (G @ S.T))

        obj = float((A - D @ S).abs().sum() + lam * S.abs().sum())
        if prev_obj is not None and abs(prev_obj - obj) <= tol * abs(prev_obj):
            break
        prev_obj = obj

    return D, S