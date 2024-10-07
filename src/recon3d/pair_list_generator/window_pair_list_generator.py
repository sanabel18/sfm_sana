from abc import ABC, abstractmethod
import json
import numpy as np

from .pair_list_generator import PairListGenerator


class WindowPairListGenerator(PairListGenerator):
    def __init__(self, config: dict):
        self.window = config['window']

    def gen_pair_list(self, id_view_list):
        pair_list = [(id_view_list[i], id_view_list[j]) for i in range(len(id_view_list)-1) for j in range(i+1, min(i+self.window+1, len(id_view_list)))]

        return pair_list