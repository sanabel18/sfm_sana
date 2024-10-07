from abc import ABC, abstractmethod
import json
import numpy as np


class PairListGenerator(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def gen_pair_list(self, id_view_list):
        pass

    @staticmethod
    def sfm_data_to_id_view_list(sfm_data_file: str):
        with open(sfm_data_file, 'r') as f:
            sfm_data = json.load(f)

        id_view_list = [view['value']['ptr_wrapper']['data']['id_view'] for view in sfm_data['views']]
        id_view_list = np.asarray(id_view_list)

        return id_view_list