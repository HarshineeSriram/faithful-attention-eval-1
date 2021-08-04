import os
import random

import numpy as np
import torch
from torch import Tensor

from auxiliary.settings import DEVICE


class WeightsEraser:

    def __init__(self):
        self.__device = DEVICE
        self.__path_to_model_dir, self.__path_to_val, self.__path_to_indices = "", "", ""
        self.__curr_filename, self.__sal_type, self.__mode = "", "", ""
        self.__fetchers = {"max": self.__indices_max, "rand": self.__indices_rand,
                           "grad": self.__indices_grad, "grad_prod": self.__indices_grad_prod}

    def set_path_to_log(self, path: str):
        self.__path_to_val = os.path.join(path, "val")
        os.makedirs(self.__path_to_val)
        self.__path_to_indices = os.path.join(path, "indices")
        os.makedirs(self.__path_to_indices)

    def set_path_to_model_dir(self, path: str):
        self.__path_to_model_dir = path

    def set_curr_filename(self, filename: str):
        self.__curr_filename = filename

    def set_saliency_type(self, saliency_type: str):
        self.__sal_type = saliency_type

    def set_erasure_mode(self, mode: str):
        self.__mode = mode

    def __load_grad(self, x: torch.Tensor) -> Tensor:
        path_to_grad = os.path.join(self.__path_to_model_dir, "grad", self.__sal_type, self.__curr_filename)
        grad = torch.from_numpy(np.load(path_to_grad))
        if x.shape != grad.shape:
            raise ValueError("Input-gradient shapes mismatch! Received input: {}, grad: {}".format(x.shape, grad.shape))
        return grad.to(self.__device)

    def __load_saliency_mask(self) -> Tensor:
        path_to_mask = os.path.join(self.__path_to_model_dir, "att", self.__sal_type, self.__curr_filename)
        return torch.from_numpy(np.load(path_to_mask, allow_pickle=True)).to(self.__device)

    def erase(self, saliency_mask: Tensor = None, n: int = 1) -> Tensor:
        """
        Zeroes out one weight in the input saliency mask according to the selected mode and logs erased value
        and corresponding index to file
        :param saliency_mask: a saliency mask scaled to the original input. If not provided, will be loaded
        :param n: the number of indices to select (upper bounded by the length of the flattened saliency mask)
        :return: the input saliency mask with an item zeroed out
        """
        if saliency_mask is None:
            saliency_mask = self.__load_saliency_mask()

        s = saliency_mask.shape
        saliency_mask = torch.flatten(saliency_mask, start_dim=1)

        indices = self.__fetch_indices(saliency_mask, n)
        val = saliency_mask[:, indices]
        saliency_mask[:, indices] = 0

        saliency_mask = saliency_mask.view(s)

        self.__log_erasure(val, indices, n)

        return saliency_mask

    def __log_erasure(self, val: Tensor, indices: Tensor, n: int):
        filename = "{}_{}_{}_{}".format(self.__curr_filename, self.__sal_type, self.__mode, n)
        np.save(os.path.join(self.__path_to_val, filename), val.detach().cpu().numpy())
        np.save(os.path.join(self.__path_to_indices, filename), indices.detach().cpu().numpy())

    def __fetch_indices(self, x: Tensor, n: int = 1) -> Tensor:
        supp_fetchers = self.__fetchers.keys()
        if self.__mode not in supp_fetchers:
            raise ValueError("Index fetcher '{}' for weights erasure not supported! Supported fetchers: {}"
                             .format(self.__mode, supp_fetchers))
        return self.__fetchers[self.__mode](x)[:, :n]

    @staticmethod
    def __indices_rand(x: Tensor) -> Tensor:
        indices = []
        for i in enumerate(range(x.shape[1])):
            item_indices = list(range(x.shape[1]))
            random.Random(i).shuffle(indices)
            indices.append(item_indices)
        return torch.LongTensor(indices)

    @staticmethod
    def __indices_max(x: Tensor) -> Tensor:
        _, indices = torch.sort(x, descending=True)
        return indices

    def __indices_grad(self, x: Tensor) -> Tensor:
        return self.__indices_max(self.__load_grad(x))

    def __indices_grad_prod(self, x: Tensor) -> Tensor:
        grad = self.__load_grad(x)
        grad_prod = grad * x
        return self.__indices_max(grad_prod)
