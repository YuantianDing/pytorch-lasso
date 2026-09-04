import warnings
import torch


def lstsq(b, A):
    """Solve min_x ||A x - b||_2 (least-norm solution when A is wide)."""
    return torch.linalg.lstsq(A, b).solution


def ridge(b, A, alpha=1e-4):
    # right-hand side
    rhs = torch.matmul(A.T, b)
    # regularized gram matrix
    M = torch.matmul(A.T, A)
    M.diagonal().add_(alpha)
    # solve
    L, info = torch.linalg.cholesky_ex(M)
    if info != 0:
        raise RuntimeError("The Gram matrix is not positive definite. "
                           "Try increasing 'alpha'.")
    x = torch.cholesky_solve(rhs, L)
    return x


def batch_cholesky_solve(b, A):
    """
    Solve a batch of PSD linear systems, with a unique matrix A_k for
    each batch entry b_k
    """
    assert b.dim() == 2  # [B,D]
    assert A.dim() == 3  # [B,D,D]
    b = b.unsqueeze(2)  # [B,D,1]
    L, info = torch.linalg.cholesky_ex(A)
    if torch.all(info == 0):
        x = torch.cholesky_solve(b, L)  # [B,D,1]
    else:
        warnings.warn('Cholesky factorization failed. Reverting to LU '
                      'decomposition...')
        x = torch.linalg.solve(A, b)  # [B,D,1]
    return x.squeeze(2)