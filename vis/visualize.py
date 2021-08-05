import matplotlib
matplotlib.use('Agg')
import os
import time
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from auxiliary.settings import DEVICE
from auxiliary.utils import scale, rescale
from classes.tasks.ccc.multiframe.data.TCC import TCC

from classes.tasks.ccc.multiframe.modules.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet
from classes.tasks.ccc.multiframe.modules.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet

# ----------------------------------------------------------------------------------------------------------------
hidden_size = 128
kernel_size = 5
model_type = "conf_tccnet"

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}
SAL_TYPES = {"att_tccnet": "spatiotemp", "conf_tccnet": "spatiotemp", "conf_att_tccnet": "spatiotemp"}
# ----------------------------------------------------------------------------------------------------------------


class Visualize:

    def __init__(self, img_list: list, model_path=None, independent_models=None, cascading_models=None,
                 this_model_type=None) -> None:

        # The number of folds to be processed (either 1, 2 or 3)
        self.NUM_FOLDS = 1

        # Where to save the generated visualizations
        # PATH_TO_SAVED = os.path.join("vis", "plots", "cc_train_binary_confidence_{}".format(time.time()))
        self.PATH_TO_SAVED = os.path.join("vis", "plots", "tmp_{}".format(time.time()))
        os.makedirs(self.PATH_TO_SAVED, exist_ok=True)

        global model_type
        model_type = this_model_type

        self.img_list = img_list
        self.model_path = None
        self.independent_models = None
        self.cascading_models = None
        self.model_names = None

        if independent_models is not None:  # In case of the independent parameter randomization test
            self.independent_models = independent_models

        if cascading_models is not None:  # In case of the cascading parameter randomization test
            self.cascading_models = cascading_models

        if model_path is not None:
            self.model_path = model_path

    def runtime(self) -> None:

        for num_fold in range(self.NUM_FOLDS):

            test_set = TCC(train=False, data_folder="fold_0")
            dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)
            print("\n *** Generating visualizations for FOLD {} *** \n".format(num_fold))
            print(" * Test set size: {}".format(len(test_set)))

            # For parameter randomization tests
            if self.independent_models is not None:
                self.generate_vis(dataloader=dataloader,
                                  model_path_list=self.independent_models)
            elif self.cascading_models is not None:
                self.generate_vis(dataloader=dataloader,
                                  model_path_list=self.cascading_models)
            # For visualizations on any single model
            elif self.model_path is not None:
                self.generate_vis(dataloader, [self.model_path])

    def generate_vis(self, dataloader, model_path_list=None) -> None:

        for model_path in tqdm(model_path_list):
            num_iter = 0  # To stop unnecessary iterations
            if num_iter < len(self.img_list):
                this_model = MODELS[model_type](hidden_size, kernel_size, SAL_TYPES[model_type])
                this_model.load(model_path)
                this_model.evaluation_mode()
                with torch.no_grad():

                    for i, (x, m, y, path_to_seq) in enumerate(dataloader):

                        x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
                        pred, spat_att, temp_att = this_model.predict(x, m, return_steps=True)

                        temp_att = np.array(temp_att.cpu())
                        if not temp_att.shape[-1] == 1:
                            temp_att = temp_att[0]

                        file_name = path_to_seq[0].split(os.sep)[-1]

                        if file_name in self.img_list:
                            masked_original = []
                            for j in range(len(spat_att)):
                                this_x = transforms.ToPILImage()(x[0, j, :].cpu()).convert("RGB")
                                scaled_attention = rescale(spat_att[np.newaxis, np.newaxis, j, 0, :],
                                                           (512, 512)).squeeze(0).permute(1, 2, 0).to(DEVICE)
                                masked_original.append(scale(
                                    F.to_tensor(this_x).to(DEVICE).permute(1, 2, 0).to(DEVICE) * scaled_attention).to(
                                    DEVICE))

                            fig = plt.figure(constrained_layout=True)
                            n_rows, n_cols = self.plt_dims(len(spat_att))
                            grid_view = fig.add_gridspec(n_rows, n_cols, width_ratios=[5 for i in range(n_rows)],
                                                         height_ratios=[5 for i in range(n_rows)],
                                                         wspace=0.4, hspace=0.5)
                            list_of_indices = self.all_indices(n_rows, n_cols)
                            frame_idx = 1
                            heatmap_idx = 1

                            for j, (indices) in enumerate(list_of_indices):

                                if j < len(spat_att):
                                    ax = fig.add_subplot(grid_view[indices[0], indices[1]])
                                    ax.imshow(masked_original[j].cpu())
                                    ax.set_title("Frame #{}".format(frame_idx), fontsize=7)
                                    ax.axis("off")
                                    frame_idx += 1

                                else:
                                    if heatmap_idx:
                                        heatmap_annotations = [this_idx + 1 for this_idx in range(len(spat_att))]
                                        ax = fig.add_subplot(grid_view[n_rows - 1, :])
                                        heatmap = ax.imshow(np.transpose(temp_att.reshape((-1, 1))), cmap='hot',
                                                            interpolation='none')
                                        ax.set_title("Temporal attention mask", fontsize=7)
                                        ax.axis("off")
                                        for this_idx in range(len(spat_att)):
                                            ax.text(x=this_idx, y=0, s=heatmap_annotations[this_idx], ha="center",
                                                    va="center", color="blue")

                                        fig.add_subplot(grid_view[n_rows - 1, :])
                                        fig.colorbar(heatmap, orientation="horizontal", shrink=2)
                                        heatmap_idx = 0

                            if self.independent_models is None and self.cascading_models is None:
                                fig.suptitle("Attention masks for {} from model {}".format(file_name, model_type))
                            else:
                                fig.suptitle(
                                    "Attention masks for {} from {}".format(file_name, model_path.split(os.sep)[-1]),
                                    fontsize=12)
                            fig.savefig(os.path.join(model_path, file_name + '.png'), dpi=200)

                            print("\n\n Figure saved successfully at ", model_path)
                            plt.clf()
                            plt.close("all")

                num_iter += 1
            else:
                break

    def plt_dims(self, number: int = 6, n_rows: int = 0, n_cols: int = 3):
        """

        @param number: number of images in a sequence
        @param n_rows: optimal number of rows (determined recursively, no need to explicitly specify)
        @param n_cols: number of images in every row for the generated visualizations
        @return: optimal number of rows and columns for visualizations
        """
        while number - n_cols >= 0:
            number -= n_cols
            n_rows += 1
            self.plt_dims(number, n_rows=n_rows)

        if number > 0:
            n_rows += 1

        return n_rows + 1, n_cols

    @staticmethod
    def all_indices(n_rows: int = 0, n_cols: int = 3) -> list:
        """

        @param n_rows: total number of rows
        @param n_cols: total number of columns
        @return: all pairs of possible (x,y) indices for specific positions of plots in each visualization
        """
        indices = []
        for i in range(0, n_rows):
            for j in range(0, n_cols):
                indices.append([i, j])
        return indices
