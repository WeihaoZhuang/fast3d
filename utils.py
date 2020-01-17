import numpy as np
import collections
from itertools import repeat
def _grad_input_padding(grad_output_size, input_size, stride, padding, kernel_size):
    """
    grad_out_out_size : like (1,3,32,32,32)
    input_size : like(1,3,32,32,32)
    kernek_size: like (3,3,3) !
    """
    input_size = list(input_size)
    k = len(grad_output_size) - 2

    if len(input_size) == k + 2:
        input_size = input_size[-k:]
    if len(input_size) != k:
        raise ValueError("input_size must have {} elements (got {})".format(k + 2, len(input_size)))

    def dim_size(d):
        return ((grad_output_size[d + 2] - 1) * stride[d] - 2 * padding[d] +
                kernel_size[d])

    min_sizes = [dim_size(d) for d in range(k)]
    max_sizes = [min_sizes[d] + stride[d] - 1 for d in range(k)]
    for size, min_size, max_size in zip(input_size, min_sizes, max_sizes):
        if size < min_size or size > max_size:
            raise ValueError(
                ("requested an input grad size of {}, but valid sizes range "
                 "from {} to {} (for a grad_output_size of {})").format(
                     input_size, min_sizes, max_sizes,
                     grad_output_size[2:]))

    return tuple(input_size[d] - min_sizes[d] for d in range(k))


def return_log_name(dict_value):
    to_log_name = lambda dict_value: str(tuple(i for i in dict_value))
    base_log = to_log_name(dict_value)
    return base_log+"_fwd.log", base_log+"_bwd_inp_grad.log", base_log+"_bwd_wei_grad.log"


def reshape_inp_weight_shape(x):
    bs,c,z,h,w = x
    return (bs*c,1,z,h,w)

def conv3d_ZHW_size(x,s,p,k,d=1):
    return int(np.floor(((x + 2 * p - d * (k - 1) - 1)) / s) + 1)


def _ntuple(n):
    """
    Copied from PyTorch source code (https://github.com/pytorch).
    """
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_triple = _ntuple(3)


