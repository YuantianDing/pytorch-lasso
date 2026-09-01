import torch


def _weighted_median(values, weights):
    """Weighted median along the last dim: the smallest v with cumulative
    weight (in sorted order) reaching half of the total."""
    order = values.argsort(dim=-1)
    sv = values.gather(-1, order)
    sw = weights.expand_as(values).gather(-1, order)
    cw = sw.cumsum(-1)
    idx = (cw >= 0.5 * cw[..., -1:]).float().argmax(-1, keepdim=True)
    return sv.gather(-1, idx).squeeze(-1)


def _pcd_chunk(x, W, z, alpha, maxiter, n_update, tol):
    # x: [N,D]  W: [D,K]  z: [N,K]
    Wt = W.T                                        # [K,D]
    absW = Wt.abs()                                 # [K,D]
    inv = torch.where(Wt != 0, 1.0 / Wt, torch.zeros_like(Wt))
    # candidate points for the 1-D subproblems; the l1 penalty adds the
    # point 0 with weight alpha
    zero_pt = x.new_zeros(z.shape[0], z.shape[1], 1)
    alpha_w = x.new_full((z.shape[1], 1), float(alpha))
    weights = torch.cat((absW, alpha_w), dim=-1)    # [K,D+1]

    r = x - torch.matmul(z, W.T)                    # [N,D]
    for _ in range(maxiter):
        # a[n,j,:] = residual with coordinate j removed
        a = r[:, None, :] + z[:, :, None] * Wt[None]          # [N,K,D]
        points = torch.cat((a * inv[None], zero_pt), dim=-1)  # [N,K,D+1]
        t = _weighted_median(points, weights)                 # [N,K]
        new = (a - t[:, :, None] * Wt[None]).abs().sum(-1) + alpha * t.abs()
        old = r.abs().sum(-1, keepdim=True) + alpha * z.abs()
        gain = old - new                                      # [N,K]
        if gain.max() <= tol:
            break
        # update the n_update coordinates with the best improvement
        top = gain.topk(min(n_update, gain.shape[1]), dim=1)
        keep = top.values > 0
        rows = torch.arange(z.shape[0], device=z.device)[:, None].expand_as(top.indices)
        z_par = z.clone()
        z_par[rows[keep], top.indices[keep]] = t[rows[keep], top.indices[keep]]
        r_par = x - torch.matmul(z_par, W.T)
        if n_update > 1:
            # coordinate optima were computed with the others fixed, so a
            # joint update may overshoot: per sample, accept it only if it
            # beats the guaranteed-monotone single best coordinate
            z_one = z.clone()
            best = top.indices[:, 0]
            z_one[rows[:, 0], best] = t[rows[:, 0], best]
            r_one = x - torch.matmul(z_one, W.T)
            f_par = r_par.abs().sum(-1) + alpha * z_par.abs().sum(-1)
            f_one = r_one.abs().sum(-1) + alpha * z_one.abs().sum(-1)
            use_par = (f_par <= f_one)[:, None]
            z = torch.where(use_par, z_par, z_one)
            r = torch.where(use_par, r_par, r_one)
        else:
            z, r = z_par, r_par
    return z


def parallel_coord_descent(x, W, z0=None, alpha=1.0, maxiter=20, n_update=1,
                           tol=1e-8, chunk_size=None):
    r"""Parallel coordinate descent for the LAD-lasso coding problem

    .. math::
        \min_z \|x - z W^T\|_1 + \alpha \|z\|_1

    Each sweep 1) solves every coordinate's 1-D subproblem exactly — a
    weighted median of the points :math:`a_i / W_{ij}` with weights
    :math:`|W_{ij}|` plus the point 0 with weight :math:`\alpha` — and
    2) applies the ``n_update`` coordinates (per sample) with the largest
    objective improvement.  ``n_update=1`` is greedy coordinate descent;
    larger values update coordinates in parallel, falling back per sample
    to the single best coordinate whenever the joint update would not
    beat it (so every sweep is monotone).

    x : Tensor of shape [n_samples, n_features]
    W : Tensor of shape [n_features, n_components]
    z0 : optional initial codes, [n_samples, n_components]
    chunk_size : samples per chunk; each sweep materializes
        chunk_size x n_components x (n_features + 1) values.
    """
    n_samples, n_features = x.shape
    n_components = W.shape[1]
    z = x.new_zeros(n_samples, n_components) if z0 is None else z0
    if chunk_size is None:
        chunk_size = max(1, int(2e8 // (n_components * (n_features + 1))))
    out = []
    for start in range(0, n_samples, chunk_size):
        sl = slice(start, start + chunk_size)
        out.append(_pcd_chunk(x[sl], W, z[sl], alpha, maxiter, n_update, tol))
    return torch.cat(out, dim=0)
