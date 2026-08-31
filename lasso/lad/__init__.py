r"""LAD-lasso dictionary learning (robust l1 fit) and companions.

Solvers for the matrix-factorization problem

.. math::
    \min_{D, S} \; \|A - D S\|_1 + \lambda \|S\|_1
    \quad s.t. \quad \|d_j\|_2 \le 1,

where columns of ``A`` (m, n) are data points, ``D`` (m, k) is the
dictionary and ``S`` (k, n) the sparse codes — note the transposed
convention relative to :mod:`lasso.linear` (``X = A.T``, ``W = D``,
``Z = S.T``).

- :func:`admm_solve` — LAD-lasso via ADMM (l1 fit *and* l1 penalty), with
  optional ``torch.compile`` acceleration and a top-k warm start that
  makes iteration 0 the k-largest-columns truncation solution.
- :func:`ista_solve` — the traditional lasso counterpart
  (:math:`\frac12\|A - DS\|_F^2 + \lambda\|S\|_1`) by alternating ISTA and
  least squares, same conventions and warm start.
- :func:`lasso.lad.lad_lasso.ista_solve` — proximal-gradient LAD-lasso on
  a smoothed (Huber / pseudo-Huber) l1 fit, kept for reference.
"""

from .lad_lasso import admm_solve
from .lasso import ista_solve

__all__ = ["admm_solve", "ista_solve"]
