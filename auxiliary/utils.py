import argparse
import functools
import os
import random
import re

import numpy as np
import torch

SEPARATOR = {"stars": "".join(["*"] * 100), "dashes": "".join(["-"] * 100), "dots": "".join(["."] * 100)}


def get_device(device_type: str) -> torch.device:
    if device_type == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(device_type))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(device_type))
        return torch.device(device_type)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(device_type))


def make_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def print_namespace(namespace: argparse.Namespace):
    print("\n" + SEPARATOR["dashes"])
    print("\n\t *** INPUT NAMESPACE PARAMETERS *** \n")
    for arg in vars(namespace):
        print(("\t - {} " + "".join(["."] * (25 - len(arg))) + " : {}").format(arg, getattr(namespace, arg)))
    print("\n" + SEPARATOR["dashes"] + "\n")


def infer_path(namespace: argparse.Namespace) -> str:
    return os.path.join(namespace.path_to_pretrained, namespace.sal_type, namespace.model_type, namespace.data_folder)


def experiment_header(title: str):
    print("\n" + SEPARATOR["stars"])
    print("\t\t {}".format(title))
    print(SEPARATOR["stars"] + "\n")


def overload(func: callable) -> any:
    def fake_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return fake_wrapper


def overloads(base: callable):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper
