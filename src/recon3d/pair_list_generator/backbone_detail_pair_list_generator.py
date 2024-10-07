import itertools

from .pair_list_generator import PairListGenerator


class BackboneDetailPairListGenerator(PairListGenerator): # for object
    def __init__(self, config: dict):
        self.backbone_frame_gap = config['backbone_frame_gap']

    def gen_pair_list(self, id_view_list):
        backbone_id_view_list = id_view_list[::self.backbone_frame_gap]
        pair_list = list(itertools.product(backbone_id_view_list, id_view_list))
        pair_list = [pair for pair in pair_list if pair[0] != pair[1]]

        return pair_list