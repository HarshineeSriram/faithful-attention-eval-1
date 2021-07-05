import os
import random
from typing import Tuple

import pandas as pd
import torch
from torch import Tensor


class WeightsEraser:

    def __init__(self, path_to_save_dir: str = None):
        self.__path_to_save_dir = path_to_save_dir

    def set_path_to_save_dir(self, path_to_save_dir: str):
        self.__path_to_save_dir = path_to_save_dir

    def single_weight_erasure(self, original_input: Tensor, saliency_mask: Tensor, mode: str) -> Tensor:
        """
        Zeroes out one weight in the input saliency mask according to the selected mode
        :param original_input: the input tensor data (e.g., image, text, encodings etc.)
        :param saliency_mask: a saliency mask scaled to the original input
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
        :return: the original input weighted by the saliency mask with an item zeroed out
        """
        val, idx, saliency_mask = WeightsEraser.erase_single_weight(saliency_mask, mode)
        log_data = pd.DataFrame({"{}_val".format(mode): val, "{}_idx".format(mode): idx})
        header = log_data.keys() if not os.path.exists(self.__path_to_save_dir) else False
        log_data.to_csv(self.__path_to_save_dir, mode='a', header=header, index=False)
        return (original_input * saliency_mask).clone()

    @staticmethod
    def erase_single_weight(saliency_mask: Tensor, mode: str = "rand") -> Tuple:
        """
        Erases one weight from the given saliency mask by zeroing it out.
        The weight is selected according to the given mode
        :param saliency_mask: a saliency mask, possibly multi-dimensional
        :param mode: the criterion to select the saliency weight to erase, it can be either:
            - "rand": a random weight in the given mask
            - "max": the highest weight in the given mask
        :return the value and corresponding tensor index to be removed and the processed saliency mask
        """
        s = saliency_mask.shape
        saliency_mask = torch.flatten(saliency_mask)
        val, idx = None, None

        if mode == "max":
            (_, max_indices) = torch.max(saliency_mask, dim=0, keepdim=True)
            idx = max_indices[random.randint(0, max_indices.shape[0] - 1)]
            saliency_mask[idx] = 0
        elif mode == "rand":
            idx = random.randint(0, saliency_mask.shape[0] - 1)
            saliency_mask[idx] = 0
        else:
            raise ValueError("Mode '{}' not supported!".format(mode))

        saliency_mask = saliency_mask.view(s)

        return val, idx, saliency_mask
