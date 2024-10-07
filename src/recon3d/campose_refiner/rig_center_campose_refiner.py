import itertools
import numpy as np
from sklearn.cluster import KMeans

from .campose_refiner import CamposeRefiner


class RigCenterCamposeRefiner(CamposeRefiner): # for object
    def __init__(self, config: dict):
        self.num_rig_views = config['num_rig_views']
        self.min_num_rig_views = self.num_rig_views // 2

    def refine(self, sfm_data):
        id_view_list, id_view_to_id_pose_dict, pose_dict = CamposeRefiner.parse_sfm_data(sfm_data)
        frame_id_view_list = np.reshape(id_view_list, (-1, self.num_rig_views))
        valid_frame_id_view_list = []
        valid_frame_center_list = []
        frame_cluster_center_list = []
        for frame_id_view in frame_id_view_list:
            valid_frame_id_view = []
            valid_frame_center = []
            for id_view in frame_id_view:
                if id_view_to_id_pose_dict[id_view] in pose_dict:
                    valid_frame_id_view.append(id_view)
                    valid_frame_center.append(pose_dict[id_view_to_id_pose_dict[id_view]]['value']['center'])
                    
            if len(valid_frame_id_view) > self.min_num_rig_views:
                valid_frame_id_view_list.append(valid_frame_id_view)
                valid_frame_center_list.append(valid_frame_center)
                frame_cluster_center_list.append(KMeans(n_clusters=2).fit(valid_frame_center).cluster_centers_)

        center_list = np.array([c for frame_center in valid_frame_center_list for c in frame_center])
        rig_dist = np.percentile(np.linalg.norm(center_list[:-1, :] - center_list[1:, :], axis=1), 30, interpolation='nearest')
        frame_cluster_center_list = np.array(frame_cluster_center_list)
        frame_cluster_center_dist_list = np.linalg.norm(frame_cluster_center_list[:, 0, :] - frame_cluster_center_list[:, 1, :], axis=1)
        is_good_frame_list = frame_cluster_center_dist_list < 3 * rig_dist

        accum_list = []
        accum = 0
        for is_good_frame in is_good_frame_list:
            if is_good_frame:
                accum += 1
            else:
                accum = 0
            accum_list.append(accum)
        longest_sec_end = np.argmax(accum_list)
        longest_sec_beg = longest_sec_end - accum_list[longest_sec_end] + 1

        if accum_list[longest_sec_end] == 0:
            # all bad center, really rare condition
            good_center_list = [np.mean(frame_center, axis=0) for frame_center in valid_frame_center_list]
        else:
            good_center_list = [np.mean(frame_center, axis=0) if is_good_frame else None for frame_center, is_good_frame in zip(valid_frame_center_list, is_good_frame_list)]
            for i in range(longest_sec_beg-1, 0-1, -1):
                if not is_good_frame_list[i]:
                    if i+2 >= len(good_center_list) or good_center_list[i+2] is None:
                        anchor = good_center_list[i+1]
                    else:
                        anchor = 2.5 * good_center_list[i+1] - 1.5 * good_center_list[i+2]
                    choice = np.argmin(np.linalg.norm(frame_cluster_center_list[i] - np.reshape(anchor, (1, -1)), axis=1))
                    good_center_list[i] = frame_cluster_center_list[i][choice]
            for i in range(longest_sec_end+1, len(good_center_list)):
                if not is_good_frame_list[i]:
                    if i-2 < 0 or good_center_list[i-2] is None:
                        anchor = good_center_list[i-1]
                    else:
                        anchor = 2.5 * good_center_list[i-1] - 1.5 * good_center_list[i-2]
                    choice = np.argmin(np.linalg.norm(frame_cluster_center_list[i] - np.reshape(anchor, (1, -1)), axis=1))
                    good_center_list[i] = frame_cluster_center_list[i][choice]

        for frame_idx, (frame_id_view, frame_center) in enumerate(zip(valid_frame_id_view_list, valid_frame_center_list)):
            if not is_good_frame_list[frame_idx]:
                for id_view, center in zip(frame_id_view, frame_center):
                    if np.linalg.norm(center - good_center_list[frame_idx]) > 3 * rig_dist:
                        pose_dict[id_view_to_id_pose_dict[id_view]]['value']['center'] = good_center_list[frame_idx].tolist()

        return sfm_data