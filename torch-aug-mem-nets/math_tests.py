import torch
from torch import Tensor
import cuda
from time_utils import timeit
import tensor_math


@timeit
def naive_cossim(memory: Tensor, read_keys: Tensor):
    """
    B=batch size; N=memory_height; M=memory_width; R=num read heads;
    :param memory: Tensor(B, N, M)
    :param read_keys: Tensor(B, R, M)
    :return: Tensor(B, R, N) -- one content based weightings per read head per batch

    This is am implementation of batched_cossim intending to produce the same output, but using pytorch's
    built in cosinsimilarity function and using loops, to confirm correctness. Useful to compare speeds and
    values obtained.
    :return:
    """
    B = memory.shape[0]
    R = read_keys.shape[1]
    N = memory.shape[1]
    out = torch.zeros(B, R, N, device=cuda.device)

    f = torch.nn.CosineSimilarity(dim=0)

    for b in range(B):
        for r in range(R):
            for n in range(N):
                row = memory[b][n]
                key = read_keys[b][r]
                out[b][r][n] = f(row, key)
    return out


def test_cossim():
    B = 1000
    N = 100
    M = 8
    R = 4
    cpu = True
    memory = torch.rand(B, N, M, device=cuda.device)
    read_keys = torch.rand(B, R, M, device=cuda.device)
    return naive_cossim(memory, read_keys), tensor_math.batched_cossim(memory, read_keys)


def test_cossim_speed():
    B = 1000
    N = 100
    M = 8
    R = 4
    cpu = True
    memory_1 = torch.rand(B, N, M, device=cuda.device)
    read_keys_1 = torch.rand(B, R, M, device=cuda.device)
    memory_2 = torch.rand(B, N, M, device=cuda.device)
    read_keys_2 = torch.rand(B, R, M, device=cuda.device)
    return naive_cossim(memory_1, read_keys_1), tensor_math.batched_cossim(memory_2, read_keys_2)


if __name__ == "__main__":
    test_cossim()
