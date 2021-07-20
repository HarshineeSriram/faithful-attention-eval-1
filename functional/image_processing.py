import math
from typing import Union, List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL.Image import Image
from scipy.spatial.distance import jensenshannon
from torch import Tensor
from torch.nn.functional import interpolate

from auxiliary.settings import DEVICE


def correct(img: Image, illuminant: Tensor) -> Image:
    """
    Corrects the color of the illuminant of a linear image based on an estimated (linear) illuminant
    @param img: a linear image
    @param illuminant: a linear illuminant
    @return: a non-linear color-corrected version of the input image
    """
    img = F.to_tensor(img).to(DEVICE)

    # Correct the image
    correction = illuminant.unsqueeze(2).unsqueeze(3) * torch.sqrt(Tensor([3])).to(DEVICE)
    corrected_img = torch.div(img, correction + 1e-10)

    # Normalize the image
    max_img = torch.max(torch.max(torch.max(corrected_img, dim=1)[0], dim=1)[0], dim=1)[0] + 1e-10
    max_img = max_img.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    normalized_img = torch.div(corrected_img, max_img)

    return F.to_pil_image(linear_to_nonlinear(normalized_img).squeeze(), mode="RGB")


def linear_to_nonlinear(img: Union[np.array, Image, Tensor], gamma: float = 2.2) -> Union[np.array, Image, Tensor]:
    if isinstance(img, np.ndarray):
        return np.power(img, (1.0 / gamma))
    if isinstance(img, Tensor):
        return torch.pow(img, 1.0 / gamma)
    return F.to_pil_image(torch.pow(F.to_tensor(img), 1.0 / gamma).squeeze(), mode="RGB")


def normalize(img: np.ndarray, n_factor: float = 65535.0) -> np.ndarray:
    """ Defaults to max_int = 65535.0"""
    return np.clip(img, 0.0, n_factor) * (1.0 / n_factor)


def rgb_to_bgr(x: np.ndarray) -> np.ndarray:
    return x[::-1]


def bgr_to_rgb(x: np.ndarray) -> np.ndarray:
    return x[:, :, ::-1]


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    """ Converts an image from height x width x channels to channels x height x width """
    return x.transpose((0, 3, 1, 2))


def scale(x: Tensor) -> Tensor:
    """ Scales all values of a tensor between 0 and 1 """
    x = x - x.min()
    x = x / x.max()
    return x


def rescale(x: Tensor, size: Tuple) -> Tensor:
    """ Rescale tensor to image size for better visualization """
    return interpolate(x, size, mode='bilinear')


def angular_error(x: Tensor, y: Tensor, safe_v: float = 0.999999) -> Tensor:
    x, y = torch.nn.functional.normalize(x, dim=1), torch.nn.functional.normalize(y, dim=1)
    dot = torch.clamp(torch.sum(x * y, dim=1), -safe_v, safe_v)
    angle = torch.acos(dot) * (180 / math.pi)
    return torch.mean(angle).item()


def tvd(pred: Tensor, label: Tensor) -> Tensor:
    """
    Total Variation Distance (TVD) is a distance measure for probability distributions
    https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
    """
    return (Tensor([0.5]) * torch.abs(pred - label)).sum()


def jsd(p: List, q: List) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2