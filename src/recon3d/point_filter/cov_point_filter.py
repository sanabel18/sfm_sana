from abc import ABC, abstractmethod
import json
import numpy as np
from sklearn.covariance import EllipticEnvelope

from .point_filter import PointFilter


class CovPointFilter(PointFilter):
    def __init__(self, config: dict):
        self.contamination = config['contamination']
    
    def filter(self, points):
        valid = EllipticEnvelope(contamination=self.contamination).fit_predict(points) == 1

        return valid