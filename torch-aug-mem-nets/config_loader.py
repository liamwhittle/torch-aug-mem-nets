"""
Takes values from dict and puts in class
"""

default_config = {

    # training
    "seq_len": 2,
    "batch_size": 2,
    "lr": 0.01,
    "project": "default_project",

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
    "memory_height": 5,
    "memory_width": 8,
    "controller_layers": 1,
    "gamma": 1,
    "num_read_heads": 3,
    "num_write_heads": 1,
    "max_shift": 1,
    "memory_init": 1e-6
}


class Config:
    def __init__(self, config=None):
        # if no config provided, use default. else, merge the  given with default to form the superset,
        # using the privded values over the default values wherever provided is provided
        if config is None:
            self.config = default_config
        else:
            self.config = {**default_config, **config}

        # stuff more concerned with training (we can ignore this stuff if we want to)
        self.seq_len = self.config["seq_len"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.project = self.config["project"]

        # task specifications
        self.task_input_size = self.config["task_input_size"]
        self.task_output_size = self.config["task_output_size"]
        self.task_input_range = self.config["task_input_range"]
        self.task_output_range = self.config["task_output_range"]

        # network architecture
        self.read_attention_mechanisms = self.config["read_attention_mechanisms"]
        self.write_attention_mechanisms = self.config["write_attention_mechanisms"]

        # network parameters
        self.memory_height = self.config["memory_height"]
        self.memory_width = self.config["memory_width"]
        self.controller_layers = self.config["controller_layers"]
        self.gamma = self.config["gamma"]
        self.num_read_heads = self.config["num_read_heads"]
        self.num_write_heads = self.config["num_write_heads"]
        self.max_shift = self.config["max_shift"]
        self.memory_init = self.config["memory_init"]

        # ~~~~~ aliases ~~~~~~~
        self.B = self.batch_size
        self.N = self.memory_height
        self.M = self.memory_width
        self.R = self.num_read_heads

    def __getitem__(self, query):
        return self.config[query]
