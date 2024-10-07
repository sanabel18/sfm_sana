#!/opt/conda/bin/python
import os, sys
import pickle 
import shutil
from project import PreprocProj, ReconProj, LocProj, ExportProj
import json, yaml
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump
from utils.exportutil import copy_2_tower
from utils.camposeutil import gen_dummy_campose
import io
from super_project import SuperProject, find_filename_w_label
from super_project_src_file import SuperProjectSrcFile
from super_project_src_tower import SuperProjectSrcTower
import argparse
from votool.scripts.export_sfm2internal import export_sfm_trajectory
from videoio.video_info import VideoInfo

def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="tower/file launch super project with tower repo or from file")
    # arguments of parser_file
    parser_file = subparsers.add_parser("file", help="launch super project from file")
    parser_tower = subparsers.add_parser("tower", help="launch super project from tower folder")
    parser_file.add_argument("-v", "--video_path", type=str, required=True, help='video path')
    parser_file.add_argument("-f", "--template_file", type=str, required=True, help='template file')
    parser_file.add_argument("-r", "--proj_dir_root", type=str, required=True, help='project root dir')
    parser_file.add_argument("-n", "--proj_name", type=str, required=True, help='project name')
    parser_file.add_argument("-s", "--ss", type=float, help='start time (sec)')
    parser_file.add_argument("-t", "--to", type=float, help='end time (sec)')
    parser_file.add_argument("-fp", "--overwrite_loc_fps_with_full_fps", action="store_true", help='overwirte loc fps task in template with full fps of video')
    parser_file.add_argument("-fd", "--use_full_duration", action="store_true", help='use full length video to estimate camera pose')
    parser_file.add_argument("-p", "--recon_fps", type=int, default=2, help='fps for reconstruction')
    parser_file.add_argument("-kf", "--key_frame_idx_file", type=str, help='key frame idx list')
    parser_file.set_defaults(func=run_sp_file)
    # arguments of parser_tower
    parser_tower.add_argument("-d", "--tower_dir", required=True, help='tower repo directory')
    parser_tower.add_argument("-v", "--video_path", type=str, required=True, help='video path')
    parser_tower.add_argument("-f", "--template_file", required=True, help='template file')
    parser_tower.add_argument("-r", "--proj_dir_root", required=True, help='project root dir')
    parser_tower.add_argument("-p", "--recon_fps", default=2, help='reconstruction fps')
    parser_tower.add_argument("-m", "--movement_type_json", required=True, help='json file to store movement type')
    parser_tower.add_argument("-fp", "--overwrite_loc_fps_with_full_fps", action="store_true", help='overwirte loc fps in template with full fps of video')
    parser_tower.add_argument("-kf", "--key_frame_idx_file", type=str, help='key frame idx list')
    parser_tower.set_defaults(func=run_sp_tower)
    args = parser.parse_args()
    args.func(args)
    

def run_sp_file(args):
    video_path = args.video_path
    template_file = args.template_file
    proj_dir_root = args.proj_dir_root
    proj_name = args.proj_name
    recon_fps = args.recon_fps
    use_full_duration = args.use_full_duration
    overwrite_loc_fps_with_full_fps = args.overwrite_loc_fps_with_full_fps
    key_frame_idx_file = args.key_frame_idx_file
    ss = args.ss
    to = args.to
    from videoio.video_info import VideoInfo
    info = VideoInfo(video_path)
    video_length = info.get_duration()
    
    if use_full_duration:
        ss = 0.0
        to = video_length
    else:
        if ss > to:
            raise ValueError('ss is bigger than to')
        elif ss < 0.0:
            raise ValueError('ss can not be smaller than zero.')
        elif to > video_length:
            raise ValueError(f'to can not be bigger than video length {video_length}')

    if overwrite_loc_fps_with_full_fps:
        loc_fps = info.get_framerate()
    else:
        loc_fps = None
    super_src = SuperProjectSrcFile(video_path, proj_dir_root, 
                                    proj_name, ss, to, recon_fps, 
                                    loc_fps=loc_fps, key_frame_idx_file=key_frame_idx_file)
    super_cfg_src_file = super_src.gen_super_src_yaml()
    SuperProject(super_cfg_src_file, template_file).run()
   
def run_sp_tower(args):
    tower_dir = args.tower_dir
    movement_type_json = args.movement_type_json
    template_file = args.template_file
    video_path = args.video_path
    proj_dir_root = args.proj_dir_root
    recon_fps = args.recon_fps
    overwrite_loc_fps_with_full_fps = args.overwrite_loc_fps_with_full_fps
    output_camera_pose_path = os.path.join(tower_dir, 'camera-pose.json')
    key_frame_idx_file = args.key_frame_idx_file
    
    with open(os.path.join(tower_dir, movement_type_json), 'r') as f:
        movement_dict = json.load(f)
    if movement_dict['movement_type'] == "stationary":
        gen_dummy_campose(output_camera_pose_path)
    else:
        if overwrite_loc_fps_with_full_fps:
            info = VideoInfo(os.path.join(tower_dir, video_path))
            loc_fps = info.get_framerate()
        else:
            loc_fps = None
     
        super_src = SuperProjectSrcTower(tower_dir,  
                                    video_path, proj_dir_root,
                                    recon_fps = recon_fps, loc_fps=loc_fps,
                                    key_frame_idx_file=key_frame_idx_file)

        super_cfg_src_file = super_src.gen_super_src_yaml()
        SuperProject(super_cfg_src_file, template_file).run() 
        export_sfm_trajectory(tower_dir, output_camera_pose_path)

if __name__ == "__main__":
    get_args()
