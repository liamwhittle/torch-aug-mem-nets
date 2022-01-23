"""
Contains training data in pytorch form for the copy task.
"""

import torch
import random
from dnc import DNC

from cuda import device
from torch.optim import RMSprop, Adam

from config_loader import default_config  

from wandb_utils import with_wandb
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_data_batch(batch_size, seq_len):
    """
    Generates a batch of training data for the copy task (can also be used in evaluation and testing)
    :param batch_size: Int (how many batches)
    :param seq_len: Int (what length of sequence do we want to practice copying?) total training size = seq_len * 2 + 1
    :return: Tensor(seq_len*2+1, batch_size, memory_width), Tensor(seq_len*2+1, batch_size, memory_width)
    """
    inputs = []
    outputs = []

    for _ in range(batch_size):
        # create tasks - last digit must be reserved for end of input/ output
        tasks = [[float(random.randint(0, 1)) for _ in range(7)] + [0] for _ in range(seq_len)]
        # input is set of copy items, end of read symbol, then empty inputs for the outputs
        _inputs = tasks + [[1. for _ in range(8)]] + [[0 for _ in range(8)] for _ in range(seq_len)]
        # outputs are reverse - empty while reads, end of symbol, then copy items
        _outputs = [[0. for _ in range(8)] for _ in range(seq_len)] + [[1. for _ in range(8)]] + tasks
        inputs.append(_inputs)
        outputs.append(_outputs)

    # put to GPU
    ins = torch.tensor(inputs, device=device)
    outs = torch.tensor(outputs, device=device)

    # transpose from (batch, seq_len, memory_height) to (seq_len, batch, memory_height)
    return ins.transpose(0, 1), outs.transpose(0, 1)


# parameters chosen specifically for the copy task
config = {
    "seq_len": 2,
    "batch_size": 10,
    "lr": 0.01,
    "project": "ntm_copy_task",
    "memory_height": 5,
    "memory_width": 8,
    "gamma": 1,
    "task_input_size": 8,
    "task_output_size": 8,
    "num_read_heads": 1,
    "num_write_heads": 1,
    "max_shift": 1,
    "memory_init": 1e-6
}

copy_task_config = {
    # training
    "seq_len": 5,
    "batch_size": 1000,
    "lr": 0.001,
    "project": "dnc_copy_task",

    # task specifications
    "task_input_size": 8,
    "task_output_size": 8,
    "task_input_range": None,
    "task_output_range": None,

    # model architecture choose the attention mechanisms. WARNING - not all combinations work.
    # default settings include all possible attention mechanisms
    "read_attention_mechanisms": ["previous_read", "content", "temporal_links", "shift"],
    "write_attention_mechanisms": ["previous_read", "content", "usage"],

    # network parameters
    "memory_height": 10,
    "memory_width": 8,
    "controller_layers": 1,
    "gamma": 1,
    "num_read_heads": 1,
    "num_write_heads": 1,
    "max_shift": 1,
    "memory_init": 1e-6
}


@with_wandb
def copy_task_retrain_batch(model, num_batches=10, _random=False):

    # get training data
    ins, outs = train_data_batch(batch_size=model.config["batch_size"], seq_len=model.config["seq_len"])

    # train num_batches batches
    for i in range(num_batches):
        if _random:
            ins, outs = train_data_batch(batch_size=model.config["batch_size"], seq_len=model.config["seq_len"])

        # do a batched update
        model.batch_update(ins, outs)

    return model


def copy_task_model(num_batches=10, _random=True):
    """
    Trains an NTM on the copy task using copy task parameters
    :param num_batches: number of times to try new batches
    :param _random:
    :return: NTM (the trained model)
    """
    model = DNC(copy_task_config)

    copy_task_retrain_batch(model, num_batches=num_batches, _random=_random)

    return model


def main():
    copy_task_model(num_batches=1000, _random=True)


if __name__ == "__main__":
    main()
