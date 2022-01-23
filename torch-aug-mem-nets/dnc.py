"""
The configurable implementation of the DNC
"""

import torch
import torch.nn as nn
import tensor_math as t_math
from config_loader import Config
from neural_networks import LSTMNetwork
from cuda import device
from torch import Tensor
from tensor_math import one_plus
from wandb_utils import use_wandb
import wandb


class DNC(nn.Module):
    def __init__(self, config):
        """
        Key tasks: init config, memory, temporal links, neural network
        :param config:
        """

        super().__init__()
        self.config = Config(config)

        # set required memory matrices
        self.memory = torch.ones(self.config.B, self.config.N, self.config.M, device=device, requires_grad=True) \
                      * self.config.memory_init

        # N x N matrix for
        self.temporal_links = torch.zeros(self.config.B, self.config.N, self.config.N, device=device,
                                          requires_grad=True)

        # previous read vectors - we need to pass these into the controller at the start of each forward pass
        self.prev_reads = torch.zeros(self.config.B, self.config.R, self.config.M, device=device, requires_grad=True)

        # values we need to keep from previous read and write head runs
        self.prev_read_weights = torch.zeros(self.config.B, self.config.R, self.config.N, device=device,
                                             requires_grad=True)
        self.prev_usage = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)
        self.prev_write_weights = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)
        self.precedence = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)

        # create the main neural network, which actually predicts the outputs (binary prediction for now)
        self.controller = LSTMNetwork(
            input_size=self.controller_in_size(),
            output_size=self.controller_out_size(),
            num_layers=self.config.controller_layers,
            batch_size=self.config.B)

        # create network which interprets the outputs of the read heads, to then be added the the controller output
        self.read_network = torch.nn.Linear(self.config.M * self.config.R, self.config.task_output_size)
        self.read_network.to(device)
        self.read_network.requires_grad_(True)

        # for training -- todo: put these in config file
        self.loss_function = nn.BCELoss()
        self.optim = torch.optim.RMSprop(self.parameters(), lr=2.5e-4, alpha=0.95, momentum=0.9)

    def batch_update(self, ins: Tensor, outs: Tensor) -> None:
        """
        Based on a set of inputs and outputs, perform a full forward pass and update weights
        :param ins: Tensor(seq_len, B, task_input_size)
        :param outs: Tensor(seq_len, B, task_output_size)
        :return: None
        """
        self.optim.zero_grad()
        self.reset()

        # calculate loss
        loss = torch.zeros(1, device=device)
        for i in range(ins.shape[0]):
            inference = self.forward(ins[i])
            loss += self.loss_function(inference, outs[i])

        print("train_loss: " + str(loss))
        if use_wandb:
            wandb.log("train_loss", loss)

        loss.backward()
        self.optim.step()

    def controller_in_size(self):
        """
        The controller input takes the task input and the previous read vectors (as currently designed)
        It receives recurrent inputs via LSTM cell state and hidden state as well
        :return: Int
        """
        return self.config.task_input_size + self.config.M * self.config.R

    def controller_out_size(self):
        """
        The controller has to output many different parameters
        of various sizes, and the number and size of these depends on the configuration. This method attempts to
        simply calculate the total size so we can construct a neural network accomodating exactly that number of outputs
        :return: Int
        """
        size = 0

        # the controller needs to be give the task output
        size += self.config.task_output_size

        # read keys + betas
        size += self.config.R * self.config.M

        # read betas
        size += self.config.R

        # free gates
        size += self.config.R

        # write key + beta + erase vector + add vector
        size += self.config.M * 3 + 1

        # allocation (interpolation) gate, write gate
        size += 2

        # read modes
        size += self.config.R * 3

        return size

    def init_memory(self):
        """
        Reset internal memory and temporal link matrix
        """
        return torch.ones(self.config.B, self.config.N, self.config.M, device=device, requires_grad=True) \
               * 0.000001

    def reset(self):
        """
        This function is called at the start of each sequence
        """
        self.memory = self.init_memory()
        self.controller.reset()
        # N x N matrix for
        self.temporal_links = torch.zeros(self.config.B, self.config.N, self.config.N, device=device,
                                          requires_grad=True)

        # previous read vectors - we need to pass these into the controller at the start of each forward pass
        self.prev_reads = torch.zeros(self.config.B, self.config.R, self.config.M, device=device, requires_grad=True)

        # values we need to keep from previous read and write head runs
        self.prev_read_weights = torch.zeros(self.config.B, self.config.R, self.config.N, device=device,
                                             requires_grad=True)
        self.prev_usage = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)
        self.prev_write_weights = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)
        self.precedence = torch.zeros(self.config.B, self.config.N, device=device, requires_grad=True)

    class ControllerOuts:
        def __init__(self, out_original, config):

            # # extract the controller outputs and perform necessary range bounding
            up_to = 0

            out = out_original.t()

            self.task_output = torch.sigmoid(out[up_to:up_to + config.task_output_size]).t()
            up_to += config.task_output_size

            self.read_keys = torch.sigmoid(out[up_to:up_to + config.R * config.M]).t().view(config.B, config.R, config.M)
            up_to += config.R * config.M

            self.read_betas = one_plus(out[up_to:up_to + config.R]).t()
            up_to += config.R

            self.free_gates = torch.sigmoid(out[up_to: up_to + config.R]).t()
            up_to += config.R

            self.write_key = torch.sigmoid(out[up_to:up_to + config.M]).t()
            up_to += config.M

            self.write_beta = one_plus(out[up_to:up_to + 1]).t()
            up_to += 1

            self.erase_vector = torch.sigmoid(out[up_to:up_to + config.M]).t()
            up_to += config.M

            self.write_vector = torch.sigmoid(out[up_to:up_to + config.M]).t()
            up_to += config.M

            self.allocation_gate = torch.sigmoid(out[up_to:up_to + 1]).t()
            up_to += 1

            self.write_gate = torch.sigmoid(out[up_to:up_to + 1]).t()
            up_to += 1

            self.read_modes = torch.nn.Softmax(dim=2)(out[up_to:up_to + config.R * 3].t().view(config.B, config.R, 3))
            up_to += config.R * 3

            self.total_len = up_to

    def forward(self, task: Tensor) -> Tensor:
        """
        The network takes in as input the previous memory (stored in the class object),
        the task input, and the previous task output.
        :param: task: Tensor(batch_size, task_input_size)
        :return: Tensor(batch_size, task_output_size)
        """

        B = self.config.B
        N = self.config.N
        R = self.config.R
        M = self.config.M

        # 1. get controller outputs

        # controller input is the concatentation of the task input and the flattened read head outputs (reads)
        controller_input = torch.cat((task, self.prev_reads.view(B, -1)), dim=1)
        c = self.ControllerOuts(self.controller.forward(controller_input), self.config)

        # 2. Perform read head attention calculations and determine reads
        content_weighting = t_math.batched_content_addressing(self.memory, c.read_keys, c.read_betas)
        forward_weights = t_math.forward_temporal_weights(self.temporal_links, self.prev_read_weights)
        backward_weights = t_math.backward_temporal_weights(self.temporal_links, self.prev_read_weights)
        read_weightings = t_math.interpolated_read_weighting(forward_weights,  # (B, R, N)
                                                             backward_weights,  # (B, R, N)
                                                             content_weighting,  # (B, R, N)
                                                             c.read_modes)  # (B, R, 3)
        reads = t_math.read_vectors(self.memory, read_weightings)  # (B, R, N)
        read_output = torch.sigmoid(self.read_network.forward(reads.view(B, -1)))  # (B, R * N) -> (B, M)

        # Perform write head attention calculations and write to memory
        retention_vectors = t_math.memory_retention_vector(self.prev_read_weights, c.free_gates)

        usage_vectors = t_math.usage_vector(retention_vectors,
                                            self.prev_usage,
                                            self.prev_write_weights)
        allocation_write_weighting = t_math.allocation_weighting_naive(usage_vectors)

        content_write_weighting = t_math.batched_content_addressing(self.memory, c.write_key.unsqueeze(1),
                                                                    c.write_beta).squeeze(1)
        # todo: implement
        write_weightings = t_math.interpolated_write_weighting(content_write_weighting,
                                                               allocation_write_weighting,
                                                               c.allocation_gate,
                                                               c.write_gate)

        # 3. perform write
        self.memory = t_math.write_to_memory(self.memory, c.erase_vector, c.write_vector,
                                             write_weightings)  # (B, N, M)

        # assign self some various previous values
        self.prev_read_weights = read_weightings
        self.prev_write_weights = write_weightings
        self.prev_reads = reads
        self.prev_usage = usage_vectors

        # update precedence
        self.precedence = t_math.precedence_weighting(self.precedence, write_weightings)
        # update the temporal link matrix
        self.temporal_links = t_math.update_link_matrix_dense(self.temporal_links, self.prev_write_weights,
                                                              self.precedence)

        # 5. (B, M) + (B, M)
        return torch.sigmoid(read_output + c.task_output)


def test():
    import cuda
    import config_loader

    model = DNC(config_loader.default_config)
    ins = torch.rand(model.config.B, model.config.task_input_size, device=cuda.device)

    return model.forward(ins)


if __name__ == "__main__":
    test()
