"""
Wrapped feedforward and LSTM networks used in the DNC.
"""

import torch
import torch.nn as nn
from torch import Tensor
from cuda import device, use_cuda

use_wandb = False


class FeedforwardNetwork(nn.Module):
    """
    Primarily used for interpreting the outputs of the read head into some sort of task output
    """
    def __init__(self, in_size, out_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.in_size = in_size
        self.out_size = out_size
        self.layers = [torch.nn.Linear(in_features=in_size, out_features=out_size).to(device) for _ in range(num_layers)]
        for layer in self.layers:
            layer.requires_grad_(True)

    def forward(self, ins):
        """
        Forward pass on the data
        :param ins:
        :return:
        """
        outs = ins
        for layer in self.layers:
            outs = torch.sigmoid(layer.forward(ins))
        return outs


class LSTMNetwork(nn.Module):
    """
    Credit to: https://github.com/loudinthecloud/pytorch-ntm for inspiration
    This network sets constant initial hidden and cell states for repeatability. i.e. we do not learn them on a task
    by task basis.
    Features: capable of arbitrary number of layers, supports batch trainingig!
    An NTM controller based on LSTM.
    """

    def __init__(self, input_size: int, output_size: int, num_layers: int, batch_size: int):
        """

        :param input_size: number of inputs per batch
        :param output_size: number of outputs per batch
        :param num_layers: how many hidden layers in the network
        :param batch_size: [1, inf]
        """
        super(LSTMNetwork, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.init_value = 0.0001

        # the actual LSTM which used or controlling the network and producing output
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=output_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.lstm.requires_grad_(True)

        if use_cuda:
            # todo: get rid of this overkill
            self.lstm = self.lstm.cuda(device)
            self.lstm.to(device)
            self.to(device)

        # The hidden state is a learned parameter
        # we pass hidden = (hidden_state, cell_state) as the hidden state tuple to the network
        self.hidden_state = torch.ones(self.num_layers, self.batch_size, self.output_size,
                                       device=device).to(device) * self.init_value
        self.cell_state = torch.ones(self.num_layers, self.batch_size, self.output_size,
                                     device=device).to(device) * self.init_value

    def reset(self):
        """
        Re-sets the hidden and cell state - to be called after a full sequence of inference.
        :return: None
        """
        self.hidden_state = torch.ones(self.num_layers, self.batch_size, self.output_size,
                                       device=device) * self.init_value
        self.cell_state = torch.ones(self.num_layers, self.batch_size, self.output_size,
                                     device=device) * self.init_value
        self.hidden_state.to(device)
        self.cell_state.to(device)

    def forward(self, x: Tensor) -> Tensor:
        """
        Updates internal state and returns output
        :param x: Tensor(batch_size, input_size)
        :return: Tensor(batch_size, output_size)
        """
        x = x.unsqueeze(1)  # add the sequence dimension (after batch dimension)
        out, (hidden_state, cell_state) = self.lstm.forward(x,
                                                            (self.hidden_state.to(device), self.cell_state.to(device)))
        self.hidden_state = hidden_state
        self.cell_state = cell_state
        return out.squeeze(1)
