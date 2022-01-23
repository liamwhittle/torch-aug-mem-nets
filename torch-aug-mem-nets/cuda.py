import torch
use_cuda = False
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
