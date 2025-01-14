from math import prod
from typing import Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn.functional import normalize

from auxiliary.utils import overloads
from classes.eval.mlp.core.LinearEncoder import LinearEncoder
from classes.tasks.ccc.multiframe.core.SaliencyTCCNet import SaliencyTCCNet
from classes.tasks.ccc.multiframe.submodules.attention.SpatialAttention import SpatialAttention
from classes.tasks.ccc.multiframe.submodules.attention.TemporalAttention import TemporalAttention
from classes.tasks.ccc.singleframe.submodules.squeezenet.SqueezeNetLoader import SqueezeNetLoader

# ---------------------------------------

INPUT_SIZE = (3, 512, 512)
ENCODING_SIZE = (31, 31)
NUM_SPAT_DIM = 512
NUM_TEMP_DIM = 128

INPUT_SIZE_SPAT = prod(INPUT_SIZE)
OUTPUT_SIZE_SPAT = prod((NUM_SPAT_DIM, prod(ENCODING_SIZE)))
INPUT_SIZE_TEMP = OUTPUT_SIZE_SPAT
OUTPUT_SIZE_TEMP = prod((NUM_TEMP_DIM, prod(ENCODING_SIZE)))


# ---------------------------------------


class LinearSaliencyTCCNet(SaliencyTCCNet):

    def __init__(self, sal_type: str, weights_mode: str, hidden_size: int = 128, kernel_size: int = 5):
        super().__init__(rnn_input_size=512, hidden_size=hidden_size, kernel_size=kernel_size, sal_type=sal_type)

        self.__weights_mode = weights_mode
        supp_weights_modes = ["imposed", "learned", "deactivate"]
        if self.__weights_mode not in supp_weights_modes:
            raise ValueError("Weights mode {} not supported! Supported modes are: {}"
                             .format(self.__weights_mode, supp_weights_modes))

        if self._is_saliency_active("spat"):
            self.spat_enc = LinearEncoder(input_size=INPUT_SIZE_SPAT, output_size=OUTPUT_SIZE_SPAT)
            if weights_mode == "learned":
                self.spat_sal = SpatialAttention(input_size=NUM_SPAT_DIM)

        if self._is_saliency_active("temp"):
            self.temp_enc = LinearEncoder(input_size=INPUT_SIZE_TEMP, output_size=OUTPUT_SIZE_TEMP)
            if weights_mode == "learned":
                self.temp_sal = TemporalAttention(features_size=NUM_TEMP_DIM, hidden_size=NUM_TEMP_DIM)

            del self.conv_lstm
            if sal_type == "temp":
                self.backbone = nn.Sequential(*list(SqueezeNetLoader().load(pretrained=True).children())[0][:12])

    def _weight_spat(self, x: Tensor, *args, **kwargs) -> Tuple:
        return x, Tensor()

    @staticmethod
    def _apply_spat_weights(x: Tensor, mask: Tensor, *args, **kwargs) -> Tensor:
        return (x * mask).clone()

    @overloads(SaliencyTCCNet._weight_temp)
    def _weight_temp(self, x: Tensor, hidden: Tensor, t: int, time_steps: int, *args, **kwargs) -> Tuple:
        return x[t, :, :, :], Tensor()

    @staticmethod
    @overloads(SaliencyTCCNet._apply_temp_weights)
    def _apply_temp_weights(x: Tensor, mask: Tensor, time_steps: int, *args, **kwargs) -> Tensor:
        out = []
        for t in range(time_steps):
            curr_mask = mask[t, :].unsqueeze(1).unsqueeze(2).unsqueeze(3)
            out.append(torch.div(torch.sum(x * curr_mask, dim=0), time_steps).unsqueeze(0))
        return torch.mean(torch.cat(out), dim=0).unsqueeze(0)

    def __learn_spat_sal(self, x: Tensor) -> Tensor:
        return self.spat_sal(x)

    def __encode_spat(self, x: Tensor, enc_shape: Tuple, w: Union[Tensor, Tuple]) -> Tensor:
        # Linear encoder
        if self._is_saliency_active("spat"):
            x = self.spat_enc(x).view(*enc_shape)

            if self.__weights_mode == "deactivate":
                return x

            spat_weights = w if self.__weights_mode == "imposed" else self.__learn_spat_sal(x)
            return self._apply_spat_weights(x, spat_weights)

        # Convolutional encoder
        x, _ = self._spat_comp(x)
        return x

    def __learn_temp_sal(self, x: Tensor, ts: int) -> Tensor:
        temp_weights = []
        for t in range(ts):
            temp_weights.append(self.temp_sal(x, x[t, :, :, :].unsqueeze(0)))
        return torch.stack(temp_weights).squeeze()

    def __encode_temp(self, x: Tensor, enc_shape: Tuple, w: Tensor, ts: int, bs: int) -> Tensor:
        # Linear encoder
        if self._is_saliency_active("temp"):
            x = self.temp_enc(x).view(*enc_shape)

            if self.__weights_mode == "deactivate":
                return x

            temp_weights = w if self.__weights_mode == "imposed" else self.__learn_temp_sal(x, ts)
            return self._apply_temp_weights(x, temp_weights, ts)

        # Recurrent encoder
        x, _ = self._temp_comp(x, bs)
        return x

    @overloads(SaliencyTCCNet.forward)
    def forward(self, x: Tensor, weights: Tensor) -> Tensor:
        batch_size, time_steps, num_channels, h, w = x.shape
        x = x.view(batch_size * time_steps, num_channels, h, w)
        spat_enc_shape = (time_steps, NUM_SPAT_DIM, *ENCODING_SIZE)
        temp_enc_shape = (time_steps, NUM_TEMP_DIM, *ENCODING_SIZE)

        if self._sal_type == "spatiotemp":
            sw, tw = weights
        else:
            sw, tw = (weights, Tensor()) if self._sal_type == "spat" else (Tensor(), weights)

        x = self.__encode_spat(x, spat_enc_shape, sw)
        x = self.__encode_temp(x, temp_enc_shape, tw, time_steps, batch_size)
        x = self.fc(x)
        pred = normalize(torch.sum(torch.sum(x, 2), 2), dim=1)

        return pred
