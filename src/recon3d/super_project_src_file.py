import os
from utils.safe_yaml import  safe_yaml_dump

class SuperProjectSrcFile():
    '''
    create input dict for super project
    Args:
    video_path: filenae of stitched video
    proj_dir_root: directory that a SfM project will be created
    proj_name: proj name
    ss: int start time of video(sec)
    to: int end time of video(sec)
    recon_fps: fps for reconstruction
    loc_fps: fps for final exported camera poses, if loc_fps is None, it won't be included in super_src_dict
    '''
    def __init__(self, video_path, proj_dir_root, proj_name,\
                 ss, to, recon_fps=2, loc_fps=None, key_frame_idx_file=None):
        self.video_path = video_path
        self.proj_dir = proj_dir_root
        self.proj_name = proj_name
        self.ss = ss
        self.to = to
        self.recon_fps = recon_fps
        self.loc_fps = loc_fps
        self.key_frame_idx_file = key_frame_idx_file
        self.super_src_dict = self.gen_super_src_dict()
    
    def gen_super_src_yaml(self):
        proj_dir = os.path.join(self.proj_dir, self.proj_name)
        os.makedirs(proj_dir, exist_ok=True)
        yaml_file = os.path.join(proj_dir, 'super_src.yaml')
        safe_yaml_dump(self.super_src_dict, yaml_file)
        return yaml_file
    
    def gen_super_src_dict(self):
        '''
        create necessary entries for input of super project
        meta: the type of project, can be SCENE_V0, OBJECT_VIDEO_V0, OBJECT_IMAGE_V0
        name: the name of folder that holds SfM project
        parent_dir: the location where SfM will be created. 
                    ex: /aaa/bbb/ccc
                    the full project path will be /aaa/bbb/ccc/name
        ss: start time of video for SfM 
        to: end time of video for SfM
        video_path: path of input video for SfM
        recon_fps: fps for reconstruction
        loc_fps: fps for final exported camera poses, if loc_fps is None, it won't be included in super_src_dict
        key_frame_idx_file: path of key frame index npy file
        '''
        super_src_dict = {}
        super_src_dict['meta'] = 'SCENE_V0'
        super_src_dict['name'] = self.proj_name
        super_src_dict['parent_dir'] = self.proj_dir
        super_src_dict['src_dir'] = os.path.dirname(self.video_path)
        super_src_dict['ss'] = self.ss
        super_src_dict['to'] = self.to
        super_src_dict['video_path'] = self.video_path
        super_src_dict['recon_fps'] = self.recon_fps
        if self.key_frame_idx_file:
            super_src_dict['key_frame_idx_file'] = self.key_frame_idx_file
        if self.loc_fps:
            super_src_dict['loc_fps'] = self.loc_fps
        return super_src_dict
