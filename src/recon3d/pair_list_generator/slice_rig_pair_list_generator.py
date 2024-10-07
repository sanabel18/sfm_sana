from abc import ABC, abstractmethod
import json
import numpy as np

from .pair_list_generator import PairListGenerator


class RigGenPair:
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def gen_in_frame_view(self, frame_id_view):
        pass
        # return pair_list

    @abstractmethod
    def gen_near_view(self, frame_id_view_1, frame_id_view_2):
        pass
        # return pair_list

    @abstractmethod
    def gen_far_view(self, frame_id_view_1, frame_id_view_2):
        pass
        # return pair_list


class RingGenPair(RigGenPair):
    def __init__(self):
        pass
    
    def gen_in_frame_view(self, frame_id_view):
        pair_list = []
        for i in range(len(frame_id_view)):
            pair_list.append((frame_id_view[i-1], frame_id_view[i]))

        return pair_list

    def gen_near_view(self, frame_id_view_1, frame_id_view_2):
        pair_list = []
        num_rig_view = len(frame_id_view_1)
        for i in range(num_rig_view):
            pair_list.append((frame_id_view_1[i-1], frame_id_view_2[i]))
            pair_list.append((frame_id_view_1[i], frame_id_view_2[i]))
            pair_list.append((frame_id_view_1[(i+1)%num_rig_view], frame_id_view_2[i]))

        return pair_list

    def gen_far_view(self, frame_id_view_1, frame_id_view_2):
        pair_list = list(itertools.product(frame_id_view_1, frame_id_view_2))

        return pair_list


class RingWindowGenPair(RigGenPair):
    def __init__(self):
        pass
    
    def gen_in_frame_view(self, frame_id_view):
        pair_list = []
        for i in range(len(frame_id_view)):
            pair_list.append((frame_id_view[i-1], frame_id_view[i]))

        return pair_list

    def gen_near_view(self, frame_id_view_1, frame_id_view_2):
        pair_list = []
        num_rig_view = len(frame_id_view_1)
        for i in range(num_rig_view):
            pair_list.append((frame_id_view_1[i-1], frame_id_view_2[i]))
            pair_list.append((frame_id_view_1[i], frame_id_view_2[i]))
            pair_list.append((frame_id_view_1[(i+1)%num_rig_view], frame_id_view_2[i]))

        return pair_list

    def gen_far_view(self, frame_id_view_1, frame_id_view_2):
        pair_list = []

        return pair_list


# class DenseCubemapWindowGenPair(RigGenPair):
#     def __init__(self):
#         self.pairing = [[4, 9, 7, 10],  [8, 5, 11, 6], [10, 7, 5, 8], [6, 11, 9, 4]]
    
#     def gen_in_frame_view(self, frame_id_view):
#         pair_list = []
#         for i, pairing in enumerate(self.pairing):
#             for p in pairing:
#                 pair_list.append((frame_id_view[i], frame_id_view[p]))

#         return pair_list

#     def gen_near_view(self, frame_id_view_1, frame_id_view_2):
#         pair_list = []
#         for i, pairing in enumerate(self.pairing):
#             pair_list.append((frame_id_view_1[i], frame_id_view_2[i]))
#             for p in pairing:
#                 pair_list.append((frame_id_view_1[i], frame_id_view_2[p]))
#                 pair_list.append((frame_id_view_1[p], frame_id_view_2[i]))
#         for p in range(4, 12):
#             pair_list.append((frame_id_view_1[p], frame_id_view_2[p]))

#         return pair_list

#     def gen_far_view(self, frame_id_view_1, frame_id_view_2):
#         pair_list = []

#         return pair_list


class SliceRigPairListGenerator(PairListGenerator):
    def __init__(self, config: dict):
        rig_type = config['rig_type']
        if rig_type == 'ring':
            self.rig_gen_pair = RingGenPair()
        elif rig_type == 'ring_window':
            self.rig_gen_pair = RingWindowGenPair()
        # elif rig_type == 'dense_cubemap_window':
        #     self.rig_gen_pair = DenseCubemapWindowGenPair()
        else:
            raise NotImplementedError
        self.num_rig_views = config['num_rig_views']
        self.near_frame = config['near_frame']

    def gen_pair_list(self, id_view_list):
        '''
        Note: May Cause Implicit Bugs !!!!
        Depends on how images are named and openMVG_main_SfMInit_ImageListing  
        Current implementation is, for instance, inta360 pro which has 6 cams rig
        => the names of images should be [frame index in %05d]_[camera index in %01d].jpg
        then openMVG_main_SfMInit_ImageListing will have the id_view_list of order, e.g.
        [ 0 for 00001_1.jpg, 1 for 00001_2.jpg, ... , 5 for 00001_6.jpg, 6 for 00002_1.jpg, 7 for 00002_2.jpg, ...]
        '''
        frame_id_view_list = np.reshape(id_view_list, (-1, self.num_rig_views)) # as the above
        pair_list = []
        # e.g. num_rig_views = 3
        # id_view_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # => frame_id_view_list = [
        #                           [0, 1, 2],
        #                           [3, 4, 5],
        #                           [6, 7, 8],
        #                           [9, 10, 11],
        #                         ]

        # in frame view
        for frame_id_view in frame_id_view_list:
            pair_list.extend(self.rig_gen_pair.gen_in_frame_view(frame_id_view))

        for i in range(len(frame_id_view_list)-1):
            for j in range(i+1,len(frame_id_view_list)):
                if j-i <= self.near_frame:
                    pair_list.extend(self.rig_gen_pair.gen_near_view(frame_id_view_list[i], frame_id_view_list[j]))
                else:
                    pair_list.extend(self.rig_gen_pair.gen_far_view(frame_id_view_list[i], frame_id_view_list[j]))

        return pair_list