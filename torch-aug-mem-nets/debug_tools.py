"""
Decorators and utilities for debugging pytorch functions
"""

import torch

CHECK_NAN = True
CHECK_RANGES = True


def check_all_nan(tensor):
    assert bool(torch.all(torch.logical_not(torch.isnan(tensor)))) and "NaN detected"


def check_all_in_range(tensor, lower_bound, upper_bound):
    return bool(torch.all(tensor >= lower_bound)) and bool(torch.all(upper_bound >= tensor))


def check_nan(func):
    """
    NaN values seem to be a common issue with memory agumented networks.
    Decorator checks if any arguments are NaN. Running in debug mode means we can check the
    stack trace interactively and isolate the problem.
    :param func: function taking tensor input s
    :return: original function with a check for nan in the arguments
    """

    if not CHECK_NAN:
        return func

    def check_nan_fun(*args):
        for arg in args:
            check_all_nan(arg)
        r = func(*args)
        return r

    return check_nan_fun


def check_range(lower: float, upper: float):
    """
    Constructs a decorator which is applied to a function returning a single Tensor.
    The returned decorator checks that every value in the returned Tesnor lies within lower and upper (inclusive)
    If not, an assertion error is thrown. This is
    :param lower: lower bound for all values of input tensor
    :param upper: upper bound for all values of input tensor
    :return: wrapped function
    """
    def decorator(function):
        if not CHECK_RANGES:
            return function

        def wrapper(*args):
            result = function(*args)
            assert check_all_in_range(result, lower, upper) \
                   and "Assertion Error: All not in Range: (" + str(lower) + ", " + str(upper) + ")"
            return result
        return wrapper
    return decorator

