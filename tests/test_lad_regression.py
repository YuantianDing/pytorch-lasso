import torch

from lasso.lad import lad_regression


def _problem(n=300, d=30, k=6, seed=0, outlier_frac=0.1):
    torch.manual_seed(seed)
    W = torch.randn(d, k)
    Z_true = torch.randn(n, k)
    X = Z_true @ W.T
    mask = torch.rand_like(X) < outlier_frac
    X = X + mask * (20.0 * torch.randn_like(X))
    return X, W, Z_true


def _lad(X, Z, W):
    return (X - Z @ W.T).abs().sum()


def test_beats_least_squares_and_recovers_truth():
    X, W, Z_true = _problem()
    Z_ls = torch.linalg.lstsq(W, X.T).solution.T
    Z = lad_regression(X, W, maxiter=500, tol=1e-8)
    assert Z.shape == Z_true.shape
    assert _lad(X, Z, W) < _lad(X, Z_ls, W)
    # LAD is robust to the sparse gross outliers; least squares is not
    assert (Z - Z_true).norm() < 1e-4 * (Z_ls - Z_true).norm()


def test_precompute_paths_agree():
    X, W, _ = _problem()
    Z_pre = lad_regression(X, W, maxiter=200, tol=1e-8, precompute=True)
    Z_iter = lad_regression(X, W, maxiter=200, tol=1e-8, precompute=False)
    torch.testing.assert_close(Z_pre, Z_iter, atol=1e-4, rtol=1e-4)


def test_default_precompute_rule_matches_forced():
    # maxiter * n_samples <= n_components -> per-iteration lstsq
    torch.manual_seed(1)
    X = torch.randn(2, 10)
    W = torch.randn(10, 30)
    Z_auto = lad_regression(X, W, maxiter=5)
    Z_iter = lad_regression(X, W, maxiter=5, precompute=False)
    torch.testing.assert_close(Z_auto, Z_iter)
    # wide W: the exact-fit min-norm solution is optimal (objective 0)
    assert _lad(X, Z_auto, W) < 1e-4


def test_warm_start_and_zero_iterations_are_sane():
    X, W, _ = _problem(n=50)
    Z1 = lad_regression(X, W, maxiter=50, tol=1e-8)
    Z2 = lad_regression(X, W, z0=Z1, maxiter=50, tol=1e-8)
    assert _lad(X, Z2, W) <= _lad(X, Z1, W) + 1e-3
    assert torch.isfinite(lad_regression(X, W, maxiter=1)).all()

