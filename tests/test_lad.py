import pytest
import torch

from lasso.lad import admm_dict_learning, dict_learning, lad_lasso_loss


def _outlier_problem(n=400, d=16, k_true=8, seed=0):
    torch.manual_seed(seed)
    W0 = torch.randn(d, k_true)
    W0 = W0 / W0.norm(dim=0, keepdim=True)
    Z0 = torch.zeros(n, k_true)
    for i in range(n):
        supp = torch.randperm(k_true)[:3]
        Z0[i, supp] = torch.randn(3)
    X = Z0 @ W0.T
    mask = torch.rand_like(X) < 0.05
    return X + mask * (5.0 * torch.randn_like(X))


def test_admm_dict_learning_shapes_and_constraint():
    X = _outlier_problem()
    W, losses = admm_dict_learning(
        X, n_components=24, alpha=0.1, steps=8, progbar=False, maxiter=30
    )
    assert W.shape == (16, 24)
    assert torch.all(W.norm(dim=0) <= 1.0 + 1e-5)
    assert losses.shape == (8,)
    assert losses[-1] <= losses[0]


def test_inner_solver_is_pluggable():
    X = _outlier_problem()
    W, losses = admm_dict_learning(
        X, n_components=24, alpha=0.1, steps=4, progbar=False, algorithm='cd'
    )
    assert torch.isfinite(losses).all()


def test_topk_init_runs_and_improves():
    X = _outlier_problem()
    W, losses = admm_dict_learning(
        X, n_components=24, alpha=0.1, steps=6, progbar=False, init='topk',
        maxiter=30,
    )
    assert torch.isfinite(losses).all()
    assert losses[-1] <= losses[0]


def test_dispatcher():
    X = _outlier_problem()
    W, losses = dict_learning(
        X, n_components=24, method='admm', alpha=0.1, steps=2, progbar=False,
        maxiter=10,
    )
    assert W.shape == (16, 24)
    with pytest.raises(ValueError):
        dict_learning(X, n_components=24, method='nope')
