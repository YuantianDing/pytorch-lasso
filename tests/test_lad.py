import torch

from lasso.lad import admm_solve, ista_solve


def _sparse_problem(m=64, n=2000, k_true=32, seed=0):
    torch.manual_seed(seed)
    D0 = torch.randn(m, k_true)
    D0 = D0 / D0.norm(dim=0, keepdim=True)
    S0 = torch.zeros(k_true, n)
    for j in range(n):
        supp = torch.randperm(k_true)[:4]
        S0[supp, j] = torch.randn(4)
    A = D0 @ S0
    mask = torch.rand_like(A) < 0.05
    return A + mask * (5.0 * torch.randn_like(A)), A, mask


def test_admm_solve_shapes_and_constraint():
    A, _, _ = _sparse_problem()
    D, S = admm_solve(A, k=48, max_iter=30, lam=0.2)
    assert D.shape == (64, 48) and S.shape == (48, 2000)
    assert torch.all(D.norm(dim=0) <= 1.0 + 1e-5)


def test_admm_topk_warm_start_reconstructs_top_columns():
    A, _, _ = _sparse_problem()
    D, S = admm_solve(A, k=48, max_iter=0, lam=0.2, init="topk")
    idx = A.abs().sum(0).argsort(descending=True)[:48]
    torch.testing.assert_close(D @ S[:, idx], A[:, idx], atol=1e-5, rtol=1e-4)


def test_ista_solve_decreases_objective():
    A, _, _ = _sparse_problem()
    torch.manual_seed(1)
    D0, S0 = ista_solve(A, k=48, max_iter=1, lam=0.05)
    torch.manual_seed(1)
    D, S = ista_solve(A, k=48, max_iter=50, lam=0.05)

    def obj(D, S):
        return float(0.5 * (A - D @ S).square().sum())

    assert obj(D, S) <= obj(D0, S0)
