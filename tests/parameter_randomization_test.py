import matplotlib
matplotlib.use('Agg')
import os
import torch

from classes.tasks.ccc.multiframe.modules.conf_tccnet.ModelConfTCCNet import ModelConfTCCNet
from classes.tasks.ccc.multiframe.modules.att_tccnet.ModelAttTCCNet import ModelAttTCCNet
from classes.tasks.ccc.multiframe.modules.conf_att_tccnet.ModelConfAttTCCNet import ModelConfAttTCCNet

from vis.visualize import Visualize

# ----------------------------------------------------------------------------------------------------------------
hidden_size = 128
kernel_size = 5
model_type = "conf_att_tccnet"

MODELS = {"att_tccnet": ModelAttTCCNet, "conf_tccnet": ModelConfTCCNet, "conf_att_tccnet": ModelConfAttTCCNet}
SAL_TYPES = {"att_tccnet": "spatiotemp", "conf_tccnet": "spatiotemp", "conf_att_tccnet": "spatiotemp"}
PATHS = {"att_tccnet": r'.\trained_models\att_tccnet\fold_0',
         "conf_tccnet": r'.\trained_models\conf_tccnet\fold_0',
         "conf_att_tccnet": r'.\trained_models\conf_att_tccnet\fold_0'}
# ----------------------------------------------------------------------------------------------------------------


class RandomParam:

    def __init__(self, path_to_model=None, path_to_destination=None, test_img_list=None,
                 fold_num: int = 0, batch_size: int = 1):

        """
        @param path_to_model: path to a single model.pth for which the cascading and
        independent randomization models need to be created
        @param path_to_destination: path to a folder that stores all cascading and
        independent randomization models
        @param test_img_list: images for which comparative visuals are required
        @param fold_num: number of fold
        @param batch_size: batch size for test loader
        """

        if test_img_list is None:
            test_img_list = ['test2.npy']

        self.independent_models = []  # To store the paths to each independent parameter randomization model
        self.cascading_models = []  # To store the paths to each cascading parameter randomization model

        self.model = MODELS[model_type](hidden_size, kernel_size, SAL_TYPES[model_type])  # Loading the original model
        self.model.load(path_to_model)
        self.model.evaluation_mode()

        self.destination = path_to_destination
        self.independent_destination = os.path.join(self.destination, 'independent_randomization')
        self.cascading_destination = os.path.join(self.destination, 'cascading_randomization')
        os.makedirs(self.destination, exist_ok=True)

        self.fold_num = fold_num
        self.batch_size = batch_size
        self.test_img_list = test_img_list
        self.original_layers_dict = self.model._network.state_dict()

    def layer_randomization(self, type_of_task: str = 'independent') -> None:

        """
        @param type_of_task: can be either 'independent' or 'cascading' for randomization type
        @return: creates folders and sub-folders containing all possible model variations
        """
        temp_dict = self.original_layers_dict.copy()

        for name in self.original_layers_dict.keys():
            if 'weight' in name:
                this_path = ""
                if type_of_task == 'independent':
                    # Resetting the state dictionary to its original values for every iteration
                    temp_dict = self.original_layers_dict.copy()
                    this_path = os.path.join(self.independent_destination, name)
                    os.makedirs(this_path, exist_ok=True)
                    self.independent_models.append(this_path)

                elif type_of_task == 'cascading':
                    this_path = os.path.join(self.cascading_destination, name)
                    os.makedirs(this_path, exist_ok=True)
                    self.cascading_models.append(this_path)

                temp_dict[name] = torch.rand(self.original_layers_dict[name].size())
                temp_model = MODELS[model_type](hidden_size, kernel_size, SAL_TYPES[model_type])
                temp_model._network.load_state_dict(temp_dict)
                temp_model.save(this_path)

    def run_pipeline(self) -> None:

        self.layer_randomization('independent')
        self.layer_randomization('cascading')

        visualizer = Visualize(self.test_img_list, independent_models=self.independent_models,
                               cascading_models=self.cascading_models, this_model_type=model_type)
        visualizer.runtime()
        print("\n\nAll processes complete!, "
              "you can view all generated visualizations in the sub-folders of: ", self.destination)


"""
if __name__ == '__main__':
    this_model = MODELS[model_type](hidden_size, kernel_size, SAL_TYPES[model_type])
    this_obj = RandomParam(PATHS[model_type], os.path.join(PATHS[model_type], 'parameter_randomization'),
                           test_img_list=['test2.npy', 'test3.npy', 'test5.npy'])
    this_obj.run_pipeline()
"""
