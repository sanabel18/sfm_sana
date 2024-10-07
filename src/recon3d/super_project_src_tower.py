import os
from utils.safe_yaml import safe_yaml_dump
from utils.src_marker import read_src_marker, get_src_marker_info 
from videoio.video_info import VideoInfo

class SuperProjectSrcTower():
    '''
    create input dict for super project 
    it will parse necessary information from tower repository 
    and feed into super project
    Args: 
    tower_dir: tower repo directory
    src_marker: filename source marker file
    video_path: filenae of stitched video
    proj_dir_root: directory that a SfM project will be created
    use_cut_video: bool, if use video from bypass commit
    recon_fps: int, fps for reconstruction
    loc_fps: int fps for final exported estimated camera poses
    '''
    def __init__(self, tower_dir, \
                 video_path, proj_dir_root, \
                 recon_fps = 2, loc_fps=None, key_frame_idx_file=None):
        self.tower_dir = tower_dir
        self.video_path = os.path.join(self.tower_dir, video_path)
        self.proj_dir = proj_dir_root
        self.proj_name = self.tower_dir.split('/')[-1]
        self.recon_fps = int(recon_fps)
        self.loc_fps = loc_fps
        self.video_length = VideoInfo(self.video_path).get_duration()
        self.key_frame_idx_file = key_frame_idx_file
        self.super_src_dict = self.gen_super_src_dict() 

    def gen_super_src_yaml(self):
        '''
        write self.super_src_dict to yaml file
        '''
        yaml_file = os.path.join(self.tower_dir, 'super_src.yaml')
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
        tower_src_dir: tower repository
        anchor_pts_num: the number of points to be considered with GPS anchor
        anchor_gps: the file that will be used as GPS anchor
        video_path: path of input video for SfM
        loc_fps: int fps of final exported estimated camera poses
        key_frame_idx_file: path of key frame index npy file
        '''
        super_src_dict = {}
        super_src_dict['meta'] = 'SCENE_V0'
        super_src_dict['name'] = self.proj_name
        super_src_dict['parent_dir'] = self.proj_dir
        
        super_src_dict['ss'] = 0
        super_src_dict['to'] = self.video_length
 
        super_src_dict['tower_src_dir'] = self.tower_dir
        super_src_dict['video_path'] = self.video_path
        super_src_dict['recon_fps'] = self.recon_fps
        if self.key_frame_idx_file:
            super_src_dict['key_frame_idx_file'] = self.key_frame_idx_file
        if self.loc_fps:
            super_src_dict['loc_fps'] = self.loc_fps
        return super_src_dict


