import torch


def _thresh_by_magnitude(theta, x):
    return torch.relu(torch.abs(x) - theta) * x.sign()

def batch_l1_proj_flat(x, z=1):
    """
    Implementation of L1 ball projection from:

    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    inspired from:

    https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246

    :param x: input data
    :param eps: l1 radius

    :return: tensor containing the projection.
    """

    # Computing the l1 norm of v
    v = torch.abs(x)
    v = v.sum(dim=1)

    # Getting the elements to project in the batch
    indexes_b = torch.nonzero(v > z).view(-1)
    if isinstance(z, torch.Tensor):
        z = z[indexes_b][:, None]
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    # If all elements are in the l1-ball, return x
    if batch_size_b == 0:
        return x

    # make the projection on l1 ball for elements outside the ball
    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1) - z) / (vv + 1)
    u = (mu - st) > 0
    if u.dtype.__str__() == "torch.bool":  # after and including torch 1.2
        rho = (~u).cumsum(dim=1).eq(0).sum(1) - 1
    else:  # before and including torch 1.1
        rho = (1 - u).cumsum(dim=1).eq(0).sum(1) - 1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    # gather all the projected batch
    proj_x = x.detach().clone()
    proj_x[indexes_b] = proj_x_b
    return proj_x


def batch_l1_proj(x, eps):
    batch_size = x.size(0)
    view = x.view(batch_size, -1)
    proj_flat = batch_l1_proj_flat(view, z=eps)
    return proj_flat.view_as(x)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    # advertorch
    assert isinstance(p, float) or isinstance(p, int)

    batch_size = x.size(0)
    norm = x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return (x.transpose(0, -1) * (1/norm)).transpose(0, -1).contiguous()