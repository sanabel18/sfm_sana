import itertools
import numpy as np

from .campose_refiner import CamposeRefiner


class RigMeanCamposeRefiner(CamposeRefiner): # for object
    def __init__(self, config: dict):
        self.num_rig_views = config['num_rig_views']

    def refine(self, sfm_data):
        id_view_list, id_view_to_id_pose_dict, pose_dict = CamposeRefiner.parse_sfm_data(sfm_data)
       
        frame_id_view_list = np.reshape(id_view_list, (-1, self.num_rig_views))

        for frame_id_view in frame_id_view_list:
            center_list = [pose_dict[id_view_to_id_pose_dict[id_view]]['value']['center'] for id_view in frame_id_view if id_view_to_id_pose_dict[id_view] in pose_dict]
            if len(center_list) > 0:
                center = np.mean(center_list, axis=0)

                for id_view in frame_id_view:
                    if id_view_to_id_pose_dict[id_view] in pose_dict:
                        pose_dict[id_view_to_id_pose_dict[id_view]]['value']['center'] = center.tolist()

        return sfm_data