from abc import ABC, abstractmethod
import json
import numpy as np


class CamposeRefiner(ABC):
    @abstractmethod
    def __init__(self, config: dict):
        pass

    @abstractmethod
    def refine(self, sfm_data):
        pass

    @staticmethod
    def parse_sfm_data(sfm_data):
        id_view_list = []
        id_view_to_id_pose_dict = dict()
        pose_dict = dict()
        for view in sfm_data['views']:
            id_view = view['value']['ptr_wrapper']['data']['id_view']
            id_pose = view['value']['ptr_wrapper']['data']['id_pose']
            id_view_list.append(id_view)
            id_view_to_id_pose_dict[id_view] = id_pose
        for extrinsic in sfm_data['extrinsics']:
            pose_dict[extrinsic['key']] = extrinsic
        
        id_view_list = np.asarray(id_view_list)
        
        return id_view_list, id_view_to_id_pose_dict, pose_dict