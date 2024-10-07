from abc import ABC, abstractmethod
import json
import numpy as np


class PointFilter(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass
    
    @abstractmethod
    def filter(self, points):
        pass
        # return valid