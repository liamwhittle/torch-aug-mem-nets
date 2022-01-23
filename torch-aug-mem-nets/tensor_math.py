"""
Pure functions for calculating attention, and read & write head operations on Pytorch Tensors

Each function takes as input one or more Tensors, and outputs exactly one Tensor, with no side effects (other
than what Pytorch does in the background for gradients and so forth).
Each function also has a doc string specifying the input and output Tensor shapes with reference to the calling
function's ConfigLoader parameters, using the following aliases:

B=batch_size
N=memory_height
M=memory_width
R=num_read_heads

Care has been taken to avoid expensive operations (maximising batched pytorch native tensor operations) and to
minimize indexing for space efficiency, however sometimes indexing is required. Currently the only function
which requires looping over the batches is the convolutional shift (attention mechanism from the original NTM).

The @check_nan and @check_range decorators can be turned off in debug_tools and when turned off,
will add 0 overhead to computation (as the decorators are returning un-modified functions),
however in debugging it has been found to be very useful to check that functions are behaving correctly and
for checking if values ever reach NaN.

Author: Liam Whittle
"""

import torch
from torch import Tensor

from pydnc import time_utils
from pydnc.debug_tools import check_nan, check_range
import math


@check_nan
@check_range(1, math.inf)
def one_plus(t: Tensor):
    """
    Soft function to bound all values to within [1, +inf]
    :param t: Tensor(*)
    :return: Tensor(*) within [1, +inf] for each value
    """
    return 1 + torch.log(1 + torch.exp(t))


@check_nan
@check_range(0, 1)
def batched_cossim(memory: Tensor, keys: Tensor) -> Tensor:
    """
    Here we implement the cosine similarity, between batches of two vectors of vectors, without using loops
    It takes the total combinations between the N and R vectors in memory and read_keys respectively: for each batch,
    for each read_key, for each memory location, the cosine similarity is calculated. Because of the various shape
    manipulations below, the whole process only requires 4 separate GPU calls, and thus takes 0.002 seconds to produce
    approximately 100 million outputs.
    :param memory: Tensor(B, N, M)
    :param keys: Tensor(B, R, M)
    :return: Tensor(B, R, N) -- one content based weightings per read head per batch
    """

    # bmm((B, R, M), (B, M, N)) -> (B, R, N)
    m_dot = torch.bmm(keys, memory.transpose(1, 2))  # returns Tensor(batch_size, num_read_heads, memory_height)

    b = keys.shape[0]
    r = keys.shape[1]

    # bmm((B * R, 1, M), (B * R, M, 1)) --> (B, R)
    read_norm = torch.sqrt(torch.bmm(
        keys.view(b * r, 1, -1), keys.view(b * r, -1, 1)).view(b, r)
                           )  # one dot product per batch per read head

    # bmm((B * R, 1, M), (B * R, M, 1)) --> (B, N)
    n = memory.shape[1]
    memory_norm = torch.sqrt(torch.bmm(
        memory.view(b * n, 1, -1), memory.view(b * n, -1, 1)).view(b, n))  # one dot product per batch per read head

    # (B, R) * (B, N) --> (B, R, N)
    denom = (read_norm * memory_norm.t().view(memory_norm.shape[1], memory_norm.shape[0], 1))\
        .transpose(0, 1).transpose(1, 2)

    # m_dot: (B, R, N) / denom: (B, R, N)
    sim = m_dot / (denom + 0.0001)
    return sim


@check_nan
@check_range(0, 1)
def batched_content_addressing(memory: Tensor, keys: Tensor, betas: Tensor) -> Tensor:
    """
    For content based read or write weightings
    :param memory: Tensor(B, N, M)
    :param keys: Tensor(B, R, M)
    :param betas: Tensor(B, R)
    :return: Tensor(B, R, N) -- one content based weightings per read head per batch
    """

    # compute batched cosine similarities: cos = Tensor(B, R, N)
    cos = batched_cossim(memory, keys)

    # take the sharpened exponential of all these values: e = Tensor(B, R)
    e = torch.exp((cos * betas.unsqueeze(-1)))

    return e / e.sum(-1).unsqueeze(-1)


@check_nan
@check_range(0, 1)
def memory_retention_vector(prev_read_weightings: Tensor, free_gates: Tensor) -> Tensor:
    """
    B=batch_size, R=num read heads, N=memory_height
    :param prev_read_weightings: Tensor(B, R, N)
    :param free_gates: Tensor(B, R)
    :return: Tensor(B, N)
    """
    B = prev_read_weightings.shape[0]
    R = prev_read_weightings.shape[1]

    # factors shape: (B, R, N)
    factors = torch.ones_like(prev_read_weightings) - free_gates.unsqueeze(-1) * prev_read_weightings

    # leveraging log laws to make this easier
    # todo: find more efficient way of doing this
    return torch.exp(torch.log(factors).sum(dim=1))


@check_nan
@check_range(0, 1)
def usage_vector(retention, prev_usage, prev_write_weighting) -> Tensor:
    """
    Calculated the usage vector as per the original DNC paper
    B=batch_size, R=num read heads, N=memory_height
    :param retention: Tensor(B, N)
    :param prev_usage: Tensor(B, N)
    :param prev_write_weighting: Tensor(B, N)
    :return: Tensor(B, N)
    """
    return (prev_usage + prev_write_weighting - prev_usage * prev_write_weighting) * retention


@check_nan
def read_vectors(memory: Tensor, weightings: Tensor) -> Tensor:
    """
    Multiplies the read weightings by the memory contents as per the original DNC and NTM papers
    :param memory: Tensor(B, N, M)
    :param weightings: Tensor(B, R, N)
    :return: Tensor(B, R, M)
    """
    return torch.bmm(weightings, memory)


@check_nan
def write_to_memory(prev_memory: Tensor, erase: Tensor, write: Tensor, weightings: Tensor) -> Tensor:
    """
    Writes to memory (returns a new memory matrix) as per the original DNC and NTM papers
    :param prev_memory: Tensor(B, N, M)
    :param erase: Tensor(B, M)
    :param write: Tensor(B, M)
    :param weightings: Tensor(B, N)
    :return: Tensor(B, N, M)
    """
    B = prev_memory.shape[0]
    N = prev_memory.shape[1]
    M = prev_memory.shape[2]

    # for each batch and write head we want the total combination matrix of the products of weights by write vector
    return prev_memory * (1 - (weightings.unsqueeze(-1) * erase.unsqueeze(1)).view(B, N, M)) + (
                weightings.unsqueeze(-1) * write.unsqueeze(1))


@check_nan
@check_range(0, 1)
def allocation_weighting_naive(usage_vectors: Tensor) -> Tensor:
    """
    todo: currently using a softmax version - should probably change to the actual paper
    This function is in the DNC paper described as requiring two expensive operations:
        - sorting the usage vector
        - taking the variable length product of sorted usage vectors

    We implement the naive version as described above. Sorting is much faster on the GPU,
    however it is also space inefficient to sort on the GPU. We tradeoff the time cost of a variable length product
    with an N^2 matrix multiply of a diagonalized version of the sorted matrix
    For reference, sorting 100,000,000 element tensor on GPU takes 0.3 seconds, and on CPU takes 12.687 seconds.
    Further
    B=batch_size, N=memory_height
    :param usage_vectors: Tensor(B, N)
    :return: Tensor(B, N)
    """
    # this isn't what the paper specified, but fk it.
    return torch.nn.Softmax(dim=1)(1 - usage_vectors)


@check_nan
@check_range(0, 1)
def write_content_addressing(memory: Tensor, write_key: Tensor, write_beta: Tensor) -> Tensor:
    """
    Leverages batched_content_addressing() to do write content addressing (where there is always only one write key)
    :param memory: Tensor(B, N, M)
    :param write_key: Tensor(B, M)
    :param write_beta: Tensor(B)
    :return: Tensor(B, N)
    """
    return batched_content_addressing(memory.unsqueeze(-1), write_key.unsqueeze(-1),
                                      write_beta.unsqueeze(-1)).squeeze(1)


@check_nan
@check_range(0, 1)
def interpolated_write_weighting(weighting_1: Tensor, weighting_2: Tensor, int_gate: Tensor, write_gate: Tensor) \
        -> Tensor:
    """
    Given two separate weighting (in the paper these are the content based and allocation based weightings),
    we interpolate by the interpolation gate (choose how much of each weighting to apply) and multiply by the write
    weighting (to guard against unnecessary writes)
    :param weighting_1: Tensor(B, N)
    :param weighting_2: Tensor(B, N)
    :param int_gate: Tensor(B, 1) -- interpolation gate
    :param write_gate: Tensor(B, 1)
    :return: Tensor(B, N)
    """
    return write_gate * (int_gate * weighting_1 + (1 - int_gate) * weighting_2)


@check_nan
@check_range(0, 1)
def precedence_weighting(previous_precedence: Tensor, write_weighting: Tensor) -> Tensor:
    """
    Calculates the new precedence weighting based on previous precendence and current write weighting
    The main idea is that if nothing was written, we keep the old precedence, otherwise we change it to the most
    recently written location.
    :param previous_precedence: Tensor(B, N)
    :param write_weighting: Tensor(B, N)
    :return: Tensor(B, N)
    """
    return (1 - torch.sum(write_weighting, dim=1)).unsqueeze(-1) * previous_precedence + write_weighting


@check_nan
@check_range(0, 1)
def update_link_matrix_dense(temporal_links: Tensor, write_weighting: Tensor, precedence: Tensor) -> Tensor:
    """
    This implementation is the O(N^2) version of the temporal link matrix - the original DNC paper
    explains the equations for this method before discussing the sparse alternative.
    :param temporal_links: Tensor(B, N, N)
    :param write_weighting: Tensor(B, N)
    :param precedence: Tensor(B, N)
    :return: Tensor(B, N, N)
    """
    # the various unsqueezes are required to perform a correct outer product / outer addition in batched form
    return (1 - (write_weighting.unsqueeze(-1) + write_weighting.unsqueeze(1)) * temporal_links \
           + write_weighting.unsqueeze(-1) * precedence.unsqueeze(1))


@check_nan
@check_range(0, 1)
def interpolated_read_weighting(weights_1, weights_2, weights_3, read_modes):
    """
    todo: indexing could be causing the issue - should split into 3 seperate read modes
    The final step in computing read weightings as per the original DNC paper
    :param weights_1: Tensor(B, R, N)
    :param weights_2: Tensor(B, R, N)
    :param weights_3: Tensor(B, R, N)
    :param read_modes: Tensor(B, R, 3)
    :return: Tensor(B, R, N)
    """
    # ok so we probably can't get away without indexing... OR CAN WE :o
    return weights_1 * read_modes[..., 0:1] \
           + weights_2 * read_modes[..., 1:2] \
           + weights_3 * read_modes[..., 2:3]


@check_nan
@check_range(0, 1)
def forward_temporal_weights(temporal_links: Tensor, prev_read_weighting: Tensor) -> Tensor:
    """
    Computes the forward weightings (based on writes that happened after a given write weighting)
    :temporal_links: Tensor(B, N, N)
    :prev_write_weights: Tensor(B, R, N)
    :return: Tensor(B, R, N)
    """
    return torch.bmm(temporal_links, prev_read_weighting.transpose(1, 2)).transpose(1, 2)


@check_nan
@check_range(0, 1)
def backward_temporal_weights(temporal_links: Tensor, prev_read_weighting: Tensor) -> Tensor:
    """
    Computes the forward weightings (based on writes that happened after a given write weighting)
    :temporal_links: Tensor(B, N, N)
    :prev_write_weights: Tensor(B, R, N)
    :return: Tensor(B, R, N)
    """
    return torch.bmm(temporal_links.transpose(1, 2), prev_read_weighting.transpose(1, 2)).transpose(1, 2)


@check_nan
@check_range(0, 1)
@time_utils.timeit
def convolutional_shift(attention: Tensor, shift: Tensor) -> Tensor:
    """
    Attention mechanism from the original NTM paper. Returns a version of the original tensor which has been shifted
    forward (for positive values) and backward (for negative values).
    :param attention: Tensor(B, N) - the weights to be convolved
    :param shift: Tensor(B, MaxShift * 2 + 1) - where each batch of length MaxShift is a normalised (softmax-ed) distribution
    :return: Tensor(B, N) - convolved weights
    """
    # this implementation is a bit looney because of how loopy it's loops are:
    # it uses loops regrettably instead of batching, and it doesn't support looping
    # via convolutions in the original paper. So basically we have loops where we want them
    # and no loops where we don't...
    # loops :(
    padding = int((shift.shape[1] - 1) / 2)
    return torch.cat(
        [torch.nn.functional.conv1d(attention[i].view(1, 1, -1),
                                    shift[i].view(1, 1, -1),
                                    padding=padding).view(1, -1)
            for i in range(attention.shape[0])])
