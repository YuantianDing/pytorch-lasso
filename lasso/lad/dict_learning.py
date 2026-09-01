from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..linear.sparse_encode import sparse_encode
from ..linear.dict_learning import (update_dict, update_dict_lstsq,
                                    update_dict_ridge)
from .sparse_code import sparse_code as lad_sparse_code


def _soft(x, t):
    """Elementwise soft-thresholding: sign(x) * max(|x| - t, 0)."""
    return x.sign() * (x.abs() - t).clamp(min=0)


def lad_lasso_loss(X, Z, weight, alpha=1.0):
    X_hat = torch.matmul(Z, weight.T)
    loss = (X - X_hat).abs().sum() + alpha * Z.abs().sum()
    return loss / X.size(0)


def dict_evaluate(X, weight, alpha, **kwargs):
    X = X.to(weight.device)
    Z = sparse_encode(X, weight, alpha, **kwargs)
    loss = lad_lasso_loss(X, Z, weight, alpha)
    return loss


def admm_dict_learning(X, n_components, alpha=1.0, rho=None, constrained=True,
                       persist=False, lambd=1e-2, steps=60, device='cpu',
                       progbar=True, init='orthogonal', return_codes=False,
                       dict_update=None, **solver_kwargs):
    r"""LAD-lasso dictionary learning via ADMM.

    Solves the robust (least-absolute-deviations) counterpart of
    :func:`lasso.linear.dict_learning`,

    .. math::
        \min_{W, Z} \|X - Z W^T\|_1 + \alpha \|Z\|_1
        \quad s.t. \quad \|w_j\|_2 \le 1,

    by splitting the fit term as ``E = X - Z W^T`` (scaled dual ``U``) and
    iterating, in the concise single-state form (``B = X - E + U``):

    .. code-block:: text

        Z = argmin_Z  (alpha/rho) ||Z||_1 + 1/2 ||Z W^T - B||_F^2
        W = argmin_{W in C}  ||Z W^T - B||_F^2
        X_hat = Z W^T
        A_U = X + B - X_hat
        B = A_U - soft(A_U - X_hat, 1/rho)

    The ``Z`` subproblem is a standard lasso solved by
    :func:`lasso.linear.sparse_encode` — pass ``algorithm=`` (and other
    solver kwargs) to choose among its methods ('ista', 'cd', 'gpsr',
    'interior-point', ...).  The ``W`` update is exactly
    :func:`lasso.linear.update_dict` (constrained) or
    :func:`lasso.linear.update_dict_ridge`.

    Parameters mirror :func:`lasso.linear.dict_learning`, plus:

    rho : float, optional
        ADMM penalty; the entrywise threshold absorbing residuals into
        ``E`` is ``1/rho``, which must sit well below the typical entry
        magnitude of X.  None (default) uses ``10 / mean(|X|)``.
    init : 'orthogonal' | 'topk'
        'orthogonal' matches :func:`lasso.linear.dict_learning`.  'topk'
        seeds the dictionary with the ``n_components`` largest samples by
        l1 norm and warm-starts the codes with the matching diagonal, so
        iteration 0 reproduces those samples exactly (the truncation
        solution); it implies ``persist`` semantics for the first step.
    dict_update : 'bcd' | 'lstsq' | 'ridge', optional
        Dictionary update, all from :mod:`lasso.linear`: 'bcd' is the
        per-atom block-coordinate :func:`update_dict` (default when
        ``constrained``), 'lstsq' the vectorized least squares + unit-ball
        projection :func:`update_dict_lstsq` (constrained, much faster for
        many atoms), 'ridge' the unconstrained :func:`update_dict_ridge`
        (default when not ``constrained``).
    return_codes : bool
        Also return the final codes ``Z``.  Unlike the l2 case they cannot
        be recovered by re-encoding ``X`` afterwards, since the ADMM codes
        are fitted against the target ``B``, not ``X``.
    """
    n_samples, n_features = X.shape
    X = X.to(device)
    if dict_update is None:
        dict_update = 'bcd' if constrained else 'ridge'
    if dict_update not in ('bcd', 'lstsq', 'ridge'):
        raise ValueError("invalid dict_update parameter '{}'.".format(dict_update))
    constrained = dict_update != 'ridge'
    if rho is None:
        rho = 10.0 / float(X.abs().mean().clamp_min(1e-12))

    weight, Z0 = _init_dictionary(X, n_components, init, constrained)

    Z = X.new_zeros(n_samples, n_components) if Z0 is None else Z0
    X_hat = torch.matmul(Z, weight.T)
    B = X - _soft(X - X_hat, 1.0 / rho)

    losses = torch.zeros(steps, device=device)
    with tqdm(total=steps, disable=not progbar) as progress_bar:
        for i in range(steps):
            # lasso subproblem for the codes, against the ADMM target B
            Z = sparse_encode(B, weight, alpha / rho, Z0, **solver_kwargs)
            losses[i] = lad_lasso_loss(X, Z, weight, alpha)
            if persist or init == 'topk':
                Z0 = Z

            # update dictionary (same updates as lasso.linear)
            if dict_update == 'bcd':
                weight = update_dict(weight, B, Z)
            elif dict_update == 'lstsq':
                weight = update_dict_lstsq(B, Z)
            else:
                weight = update_dict_ridge(B, Z, lambd=lambd)

            # refresh the ADMM target (E and U updates, folded)
            X_hat = torch.matmul(Z, weight.T)
            A_U = X + B - X_hat
            B = A_U - _soft(A_U - X_hat, 1.0 / rho)

            progress_bar.set_postfix(loss=losses[i].item())
            progress_bar.update(1)

    if return_codes:
        return weight, Z, losses
    return weight, losses


def _init_dictionary(X, n_components, init, constrained):
    """Shared dictionary / code initialization ('orthogonal' or 'topk')."""
    n_samples, n_features = X.shape
    Z0 = None
    if init == 'topk':
        idx = X.abs().sum(dim=1).argsort(descending=True)[:n_components]
        weight = X[idx].T.clone()
        norms = weight.norm(dim=0).clamp_min(1e-8)
        Z0 = X.new_zeros(n_samples, n_components)
        Z0[idx, torch.arange(len(idx), device=X.device)] = norms
    else:
        weight = torch.empty(n_features, n_components, device=X.device)
        nn.init.orthogonal_(weight)
    if constrained:
        weight = F.normalize(weight, dim=0)
    return weight, Z0


def alt_dict_learning(X, n_components, alpha=1.0, constrained=True, steps=60,
                      device='cpu', progbar=True, init='orthogonal',
                      return_codes=False, algorithm='pcd', **solver_kwargs):
    r"""LAD-lasso dictionary learning by alternating LAD sparse coding.

    Both half-steps are solved by the same :func:`lasso.lad.sparse_code`
    method (``algorithm=``): the codes as

    .. math::
        Z = \arg\min_Z \|X - Z W^T\|_1 + \alpha \|Z\|_1 ,

    and the dictionary as the *transposed* coding problem with no l1
    penalty, :math:`W = \arg\min_W \|X^T - W Z^T\|_1` (each atom is
    then projected onto the unit ball when ``constrained``).  No ADMM
    splitting: the l1 fit is handled directly by the coding solver.
    Parameters mirror :func:`admm_dict_learning`.
    """
    X = X.to(device)
    weight, Z = _init_dictionary(X, n_components, init, constrained)
    if Z is None:
        Z = X.new_zeros(X.shape[0], n_components)

    losses = torch.zeros(steps, device=device)
    with tqdm(total=steps, disable=not progbar) as progress_bar:
        for i in range(steps):
            Z = lad_sparse_code(X, weight, alpha, Z, algorithm=algorithm,
                                **solver_kwargs)
            losses[i] = lad_lasso_loss(X, Z, weight, alpha)
            weight = lad_sparse_code(X.T, Z, 0.0, weight, algorithm=algorithm,
                                     **solver_kwargs)
            if constrained:
                weight = weight / weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            progress_bar.set_postfix(loss=losses[i].item())
            progress_bar.update(1)

    if return_codes:
        return weight, Z, losses
    return weight, losses


_learning_methods = {
    'admm': admm_dict_learning,
    'alt': alt_dict_learning,
    # room for future LAD dictionary-learning methods, e.g. smoothed
    # (Huber) proximal-gradient or IRLS variants.
}


def dict_learning(X, n_components, method='admm', **kwargs):
    """Dispatch to a LAD-lasso dictionary-learning method.

    'admm' (:func:`admm_dict_learning`) and 'alt'
    (:func:`alt_dict_learning`) are registered; the registry reserves
    space for other methods.
    """
    if method not in _learning_methods:
        raise ValueError("invalid method parameter '{}'.".format(method))
    return _learning_methods[method](X, n_components, **kwargs)
