import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
from typing import List, Optional


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

@torch.jit.script
def sign(x):
    return torch.sign(x.sign() + 0.5)

@torch.jit.script
def tanh(x):
    return x.clamp(-15, 15).tanh()


@torch.jit.script
def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

@torch.jit.script
def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)

@torch.jit.script
def list_range(end: int):
    res: List[int] = []
    for d in range(end):
        res.append(d)
    return res

@torch.jit.script
def sabs(x, eps: float = 1e-15):
    return x.abs().add_(eps)

@torch.jit.script
def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)

@torch.jit.script
def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
            + 1382 / 155925 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x ** 3
    elif order == 2:
        return x + 1 / 3 * k * x ** 3 + 2 / 15 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x ** 3
            + 2 / 15 * k ** 2 * x ** 5
            + 17 / 315 * k ** 3 * x ** 7
            + 62 / 2835 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

def weighted_midpoint(
    xs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    k: torch.Tensor,
    reducedim: Optional[List[int]] = None,
    dim: int = -1,
    keepdim: bool = False,
    lincomb: bool = False,
    posweight: bool = False,
):
    r"""
    Compute weighted Möbius gyromidpoint.

    The weighted Möbius gyromidpoint of a set of points
    :math:`x_1,...,x_n` according to weights
    :math:`\alpha_1,...,\alpha_n` is computed as follows:

    The weighted Möbius gyromidpoint is computed as follows

    .. math::

        m_{\kappa}(x_1,\ldots,x_n,\alpha_1,\ldots,\alpha_n)
        =
        \frac{1}{2}
        \otimes_\kappa
        \left(
        \sum_{i=1}^n
        \frac{
        \alpha_i\lambda_{x_i}^\kappa
        }{
        \sum_{j=1}^n\alpha_j(\lambda_{x_j}^\kappa-1)
        }
        x_i
        \right)

    where the weights :math:`\alpha_1,...,\alpha_n` do not necessarily need
    to sum to 1 (only their relative weight matters). Note that this formula
    also requires to choose between the midpoint and its antipode for
    :math:`\kappa > 0`.

    Parameters
    ----------
    xs : tensor
        points on poincare ball
    weights : tensor
        weights for averaging (make sure they broadcast correctly and manifold dimension is skipped)
    reducedim : int|list|tuple
        reduce dimension
    dim : int
        dimension to calculate conformal and Lorenz factors
    k : tensor
        constant sectional curvature
    keepdim : bool
        retain the last dim? (default: false)
    lincomb : bool
        linear combination implementation
    posweight : bool
        make all weights positive. Negative weight will weight antipode of entry with positive weight instead.
        This will give experimentally better numerics and nice interpolation
        properties for linear combination and averaging

    Returns
    -------
    tensor
        Einstein midpoint in poincare coordinates
    """
    return _weighted_midpoint(
        xs=xs,
        k=k,
        weights=weights,
        reducedim=reducedim,
        dim=dim,
        keepdim=keepdim,
        lincomb=lincomb,
        posweight=posweight,
    )

@torch.jit.script
def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)

@torch.jit.script
def _mobius_add(x: torch.Tensor, y: torch.Tensor, k: torch.Tensor, dim: int = -1):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k ** 2 * x2 * y2
    # minimize denom (omit K to simplify th notation)
    # 1)
    # {d(denom)/d(x) = 2 y + 2x * <y, y> = 0
    # {d(denom)/d(y) = 2 x + 2y * <x, x> = 0
    # 2)
    # {y + x * <y, y> = 0
    # {x + y * <x, x> = 0
    # 3)
    # {- y/<y, y> = x
    # {- x/<x, x> = y
    # 4)
    # minimum = 1 - 2 <y, y>/<y, y> + <y, y>/<y, y> = 0
    return num / denom.clamp_min(1e-15)

@torch.jit.script
def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
            - 1 / 11 * k ** 5 * x ** 11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x ** 3
    elif order == 2:
        return x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5
    elif order == 3:
        return (
            x - 1 / 3 * k * x ** 3 + 1 / 5 * k ** 2 * x ** 5 - 1 / 7 * k ** 3 * x ** 7
        )
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x ** 3
            + 1 / 5 * k ** 2 * x ** 5
            - 1 / 7 * k ** 3 * x ** 7
            + 1 / 9 * k ** 4 * x ** 9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

@torch.jit.script
def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)

@torch.jit.script
def _dist(
    x: torch.Tensor,
    y: torch.Tensor,
    k: torch.Tensor,
    keepdim: bool = False,
    dim: int = -1,
):
    return 2.0 * artan_k(
        _mobius_add(-x, y, k, dim=dim).norm(dim=dim, p=2, keepdim=keepdim), k
    )

@torch.jit.script
def drop_dims(tensor: torch.Tensor, dims: List[int]):
    # Workaround to drop several dims in :func:`torch.squeeze`.
    seen: int = 0
    for d in dims:
        tensor = tensor.squeeze(d - seen)
        seen += 1
    return tensor

@torch.jit.script
def _geodesic_unit(
    t: torch.Tensor,
    x: torch.Tensor,
    u: torch.Tensor,
    k: torch.Tensor,
    dim: int = -1,
):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    second_term = tan_k(t / 2.0, k) * (u / u_norm)
    gamma_1 = _mobius_add(x, second_term, k, dim=dim)
    return gamma_1

@torch.jit.script
def _lambda_x(x: torch.Tensor, k: torch.Tensor, keepdim: bool = False, dim: int = -1):
    return 2 / (1 + k * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min(1e-15)



@torch.jit.script
def _mobius_scalar_mul(
    r: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1
):
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = tan_k(r * artan_k(x_norm, k), k) * (x / x_norm)
    return res_c

@torch.jit.script
def _antipode(x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    # NOTE: implementation that uses stereographic projections seems to be less accurate
    # sproj(-inv_sproj(x))
    if torch.all(k.le(0)):
        return -x
    v = x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-15)
    R = sabs(k).sqrt().reciprocal()
    pi = 3.141592653589793

    a = _geodesic_unit(pi * R, x, v, k, dim=dim)
    return torch.where(k.gt(0), a, -x)

@torch.jit.script
def _weighted_midpoint(
    xs: torch.Tensor,
    k: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    reducedim: Optional[List[int]] = None,
    dim: int = -1,
    keepdim: bool = False,
    lincomb: bool = False,
    posweight: bool = False,
):
    if reducedim is None:
        reducedim = list_range(xs.dim())
        reducedim.pop(dim)
    gamma = _lambda_x(xs, k=k, dim=dim, keepdim=True)
    if weights is None:
        weights = torch.tensor(1.0, dtype=xs.dtype, device=xs.device)
    else:
        weights = weights.unsqueeze(dim)
    if posweight and weights.lt(0).any():
        xs = torch.where(weights.lt(0), _antipode(xs, k=k, dim=dim), xs)
        weights = weights.abs()
    denominator = ((gamma - 1) * weights).sum(reducedim, keepdim=True)
    nominator = (gamma * weights * xs).sum(reducedim, keepdim=True)
    two_mean = nominator / clamp_abs(denominator, 1e-10)
    a_mean = _mobius_scalar_mul(
        torch.tensor(0.5, dtype=xs.dtype, device=xs.device), two_mean, k=k, dim=dim
    )
    if torch.any(k.gt(0)):
        # check antipode
        b_mean = _antipode(a_mean, k, dim=dim)
        a_dist = _dist(a_mean, xs, k=k, keepdim=True, dim=dim).sum(
            reducedim, keepdim=True
        )
        b_dist = _dist(b_mean, xs, k=k, keepdim=True, dim=dim).sum(
            reducedim, keepdim=True
        )
        better = k.gt(0) & (b_dist < a_dist)
        a_mean = torch.where(better, b_mean, a_mean)
    if lincomb:
        if weights.numel() == 1:
            alpha = weights.clone()
            for d in reducedim:
                alpha *= xs.size(d)
        else:
            weights, _ = torch.broadcast_tensors(weights, gamma)
            alpha = weights.sum(reducedim, keepdim=True)
        a_mean = _mobius_scalar_mul(alpha, a_mean, k=k, dim=dim)
    if not keepdim:
        a_mean = drop_dims(a_mean, reducedim)
    return a_mean