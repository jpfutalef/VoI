"""
Iterative method to guide GBM construction using VoI, coupled with some optimization algorithm.

Author: Juan-Pablo Futalef
"""
from pathlib import Path
from dataclasses import dataclass

from greyboxmodels.construction.GreyBoxRepository import GreyBoxRepository
from greyboxmodels.modelbuild import Input

@dataclass
class ConstructorParameters:
    model: object
    data: object
    voi: object
    optimizer: object

class Dataset:
    def __init__(self,
                 data):
        self.data = data

    @classmethod
    def load(cls, data):
        return

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class GreyBoxModelConstructor:
    def __init__(self,
                 model_repository: GreyBoxRepository,
                 gt_data: Dataset,
                 params: ConstructorParameters,
                 ):
        """
        This constructor enables the iterative selection of the best model from a repository of models
        and a dataset of ground truth data. The selection is guided by the Value of Information (VoI) metric.

        :param model_repository: A repository of grey-box models
        :param gt_data: Ground truth data
        :param params: Constructor parameters
        """
        self.repository = model_repository
        self.gt_data = gt_data
        self._params = params

    def construct(self):
        self.current_model = self.model
        self.current_data = self.data
        self.current_voi = self.voi(self.current_model, self.current_data)
        self.current_score = self.current_voi.score()
        while not self.optimizer.should_stop(self.current_voi):
            self.current_model = self.optimizer.optimize(self.current_voi)
            self.current_voi = self.voi(self.current_model, self.current_data)
            self.current_score = self.current_voi.score()
        return self.current_model, self.current_voi, self.current_score

    def optimize(self):
        self.current_model = self.optimizer.optimize(self.current_voi)
        self.current_voi = self.voi(self.current_model, self.current_data)
        self.current_score = self.current_voi.score()
        return self.current_model, self.current_voi, self.current_score

    def save(self, path):
        pass
