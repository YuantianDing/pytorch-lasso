import torch

from .solvers import parallel_coord_descent

_init_defaults = {
    'pcd': 'zero',
}


def initialize_code(x, weight, mode):
    n_samples = x.size(0)
    n_components = weight.size(1)
    if mode == 'zero':
        return x.new_zeros(n_samples, n_components)
    if mode == 'transpose':
        return torch.matmul(x, weight)
    raise ValueError("invalid init parameter '{}'.".format(mode))


def sparse_code(x, weight, alpha=1.0, z0=None, algorithm='pcd', init=None,
                **kwargs):
    r"""LAD-lasso sparse coding with a fixed dictionary,

    .. math::
        \min_z \|x - z W^T\|_1 + \alpha \|z\|_1 ,

    the l1-fit counterpart of :func:`lasso.linear.sparse_encode` (same
    conventions).  Algorithms: 'pcd' (parallel coordinate descent,
    :func:`lasso.lad.solvers.parallel_coord_descent`); the dispatch
    reserves space for others.
    """
    n_samples = x.size(0)
    n_components = weight.size(1)

    if z0 is not None:
        assert z0.shape == (n_samples, n_components)
    else:
        if init is None:
            init = _init_defaults.get(algorithm, 'zero')
        z0 = initialize_code(x, weight, mode=init)

    if algorithm == 'pcd':
        z = parallel_coord_descent(x, weight, z0, alpha, **kwargs)
    else:
        raise ValueError("invalid algorithm parameter '{}'.".format(algorithm))

    return z
