import wandb

"""
Weights and biases wrappers and decorators for pytorch integration
"""

use_wandb = False
logging_freq = 100


def with_wandb(function):
    """
    This is the main decorator which can be placed over a generic training pipeline function. It's purpose is
    to take care of all the weights and biases boilerplate in a standard way so we don't have to clutter the pytorch
    code with it.
    Usage: @with_wandb / model_pipeline(hyper-params)
    The argument in args[0] should be a hyper-parameters dictionary containing at least the key "project" (project name)
    :return: function wrapped by wandb init
    """

    if not use_wandb:
        return function

    def decorator(*args):
        # login to wandb if haven't already
        with wandb.init(project="copy_task", config=args[0]):
            # do the ML
            return function(args[0])

    return decorator


def log_function(function):
    """
    todo: implement
    Wraps functions outputting tensors such that the values are periodically logged
    :return: function wrapped with a wandb logger
    """
    if not use_wandb:
        return function

    def decorator(*args):
        pass

    return decorator
