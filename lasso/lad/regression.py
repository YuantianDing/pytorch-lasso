r"""Least-absolute-deviations (LAD) regression by ADMM.

.. math::
    \min_Z \; \|X - Z W^T\|_1

Same conventions as the rest of :mod:`lasso.lad` (rows of ``X`` are
samples): ``X`` is ``[n_samples, n_features]``, ``W`` is
``[n_features, n_components]`` and ``Z`` is ``[n_samples, n_components]``.
In the transposed "column" notation :math:`\min_Z \|X - W Z\|_1` this is
the same problem with ``X`` and ``Z`` transposed.
"""
import torch


def _soft(x, t):
    """Elementwise soft-thresholding: sign(x) * max(|x| - t, 0)."""
    return x.sign() * (x.abs() - t).clamp(min=0)


def lad_regression(x, weight, z0=None, rho=None, maxiter=100, tol=1e-6,
                   precompute=None):
    r"""LAD regression :math:`\min_Z \|X - Z W^T\|_1` via ADMM.

    The fit residual is split off as ``R = X - Z W^T`` (scaled dual ``U``)
    and, with ``B = X - R - U``, each iteration is

    .. code-block:: text

        Z  = argmin_Z ||Z W^T - B||_F^2          (least squares)
        R0 = X - Z W^T - U
        R  = soft(R0, 1/rho)
        U  = R - R0

    The least-squares step is a plain :func:`torch.linalg.lstsq` solve.
    When it will be applied to more right-hand sides overall than it takes
    to form the pseudo-inverse, ``maxiter * n_samples > n_components``,
    the matrix ``W (W^T W)^{-1}`` (``[n_features, n_components]``; for a
    wide ``W`` its min-norm counterpart) is computed once with
    ``torch.linalg.lstsq`` and every iteration reduces to the product
    ``Z = B @ W (W^T W)^{-1}``; otherwise ``torch.linalg.lstsq`` is called
    per iteration.  ``precompute`` overrides that choice.

    Parameters
    ----------
    x : Tensor of shape [n_samples, n_features]
    weight : Tensor of shape [n_features, n_components]
    z0 : optional warm start, [n_samples, n_components].  ``None`` starts
        from ``R = U = 0``, so the first ``Z`` is the least-squares fit.
    rho : float, optional
        ADMM penalty; the entrywise threshold absorbing residuals into
        ``R`` is ``1/rho``, which should sit well below the typical entry
        magnitude of ``x``.  ``None`` uses ``10 / mean(|x|)``, as in
        :func:`lasso.lad.admm_dict_learning`.
    maxiter : int
        Maximum number of ADMM iterations.
    tol : float
        Relative tolerance on the primal (``||R + Z W^T - X||``) and dual
        (``rho ||(R - R_prev) W||``) residuals.
    precompute : bool, optional
        Force (``True``) or forbid (``False``) precomputing the
        pseudo-inverse of ``W``; ``None`` decides by the rule above.

    Returns
    -------
    z : Tensor of shape [n_samples, n_components]
    """
    n_samples, n_features = x.shape
    n_components = weight.shape[1]
    if rho is None:
        rho = 10.0 / float(x.abs().mean().clamp_min(1e-12))
    thresh = 1.0 / rho
    if precompute is None:
        precompute = maxiter * n_samples > n_components

    if precompute:
        # W (W^T W)^{-1} = argmin_P ||W^T P - I||  (min-norm solution when
        # W is tall; the least-squares one, W^T (W W^T)^{-1}, when wide)
        eye = torch.eye(n_components, device=x.device, dtype=x.dtype)
        pinv_t = torch.linalg.lstsq(weight.T, eye).solution  # [D, K]

        def solve(b):
            return torch.matmul(b, pinv_t)
    else:
        def solve(b):
            return torch.linalg.lstsq(weight, b.T).solution.T

    if z0 is None:
        R = torch.zeros_like(x)
        U = torch.zeros_like(x)
    else:
        assert z0.shape == (n_samples, n_components)
        R0 = x - torch.matmul(z0, weight.T)
        R = _soft(R0, thresh)
        U = R - R0

    x_norm = float(x.norm().clamp_min(1e-12))
    for _ in range(maxiter):
        B = x - R - U
        z = solve(B)
        R0 = x - torch.matmul(z, weight.T) - U
        R_prev, U_prev = R, U
        R = _soft(R0, thresh)
        U = R - R0
        # primal residual R + Z W^T - X == U - U_prev; dual rho (R - R_prev) W
        primal = (U - U_prev).norm()
        dual = torch.matmul(R - R_prev, weight).norm()
        dual_scale = torch.matmul(U, weight).norm().clamp_min(1e-12)
        if primal <= tol * x_norm and dual <= tol * dual_scale:
            break
    return z
