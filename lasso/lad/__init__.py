r"""LAD-lasso: dictionary learning with a robust l1 fit term.

.. math::
    \min_{W, Z} \; \|X - Z W^T\|_1 + \alpha \|Z\|_1
    \quad s.t. \quad \|w_j\|_2 \le 1

Same conventions as :mod:`lasso.linear` (rows of ``X`` are samples).  The
outer ADMM loop handles the l1 fit; the lasso subproblem for ``Z`` is
solved by any :func:`lasso.linear.sparse_encode` algorithm, and the
dictionary update reuses :func:`lasso.linear.update_dict` /
:func:`update_dict_ridge`.  :func:`lad_regression` solves the plain
(unpenalized) LAD regression ``min_Z ||X - Z W^T||_1`` by ADMM.
"""

from . import solvers
from .dict_learning import (dict_learning, dict_evaluate, admm_dict_learning,
                            alt_dict_learning, lad_lasso_loss)
from .sparse_code import sparse_code, initialize_code
from .regression import lad_regression
