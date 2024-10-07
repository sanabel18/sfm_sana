import os
from super_project import SuperProject
import yaml, json
from utils.safe_yaml import safe_yaml_load
import glob
from super_project_src_tower import SuperProjectSrcTower
from utils.camposeutil import gen_dummy_campose
from votool.scripts.export_sfm2internal import export_sfm_trajectory
from videoio.video_info import VideoInfo

AUTO_PIPE_CONFIG = '../config/auto_pipe_config_stepsize.yaml'
CAM_POSE_NAME = 'camera-pose.json'

def decode_env(env_var):
    '''
    decode environment variable from tower auto pipeline
    example of input string: '['a','b','c']'
    Args: string
    Return: list of string
    '''
    var_list = env_var.replace("[","").replace("]","").split(",")
    var_str_list = []
    for var in var_list:
        var = var.replace("\'","")
        var_str_list.append(var)
    return var_str_list

def gen_tower_operations(tower_operations_yaml, tower_dir):
    '''
    set label/unlabel files for tower auto pipline
    write to a yaml file at directory labels_yaml
    
    It will go through folders starts with 'data_*' in tower repo
    and add each files within the folder. 
    To add those files one by one into labels.yaml is a workaround 
    since tower auto pipe does not support adding a folder into tower repo.
    
    Those files are results from SfM pipeline, should be kept for furture use
    but will not dedicate to any tower file labels, thus we
    add @dummy label and unlabel it. 

    Args:
    labels_yaml: str, path of labels.yaml
    tower_dir: str, path of tower_dir
    '''
    tower_op_dict = {"operations": []}
    
    cwd = os.getcwd()
    os.chdir(tower_dir)
    data_list  = glob.glob('data_*')
    
    add_file_list = []
    for data_dir in sorted(data_list):
        file_list = os.listdir(data_dir)
        for f in file_list:
            file_path = os.path.join(data_dir, f)
            add_file_list.append(file_path)
    
    remove_dict = {
        "remove-files-by-labels": {"labels": ['@video']}
    }
    tower_op_dict["operations"].append(remove_dict)
    add_file_list.append(CAM_POSE_NAME)
    add_dict = {"add": {"files": add_file_list}}
    tower_op_dict['operations'].append(add_dict)
    
    label_dict = {
            "label-files" :{
                "files": [CAM_POSE_NAME],
                "labels": ['@camera-pose']
                }
            }
    tower_op_dict['operations'].append(label_dict)

    with open(tower_operations_yaml, 'w') as f:
        yaml.dump(tower_op_dict, f, default_flow_style=False)
    os.chdir(cwd)

if __name__ == "__main__":
    '''
    This script will be used to activate super project by 
    using tower autopipeline. It wil create log that 
    will be stored at LOGGER_DIR

    Tower autopipeline will clone tower repo to local
    the example location looks like:
    /ailabs/tower/615c0211925b7c001344ed41
    the string '615c0211925b7c001344ed41' is commit ID 
    of that tower repository
    
    The SfM project will be created at /root/proj/sfm_proj

    Args:
    tower_repo: tower repo directory
    video: name of video file with tower label @stitched video
    movement_type: name of movement-type.json with tower label @movement-type
    node_name: the name of node the pod runing with
    pod_name: the name of the running pod
    labels: where the labels.yaml should be stored 
            ex: /aa/bb/labels.yaml
    '''
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--tower_repo", required=True, help='tower repo directory')
    parser.add_argument("-v", "--video", required=True, help='stitched video: name of video file with tower label @stitched video')
    parser.add_argument("-m", "--movement_type", required=True, help='movement type json: name of video_movement_type.json with tower label @movement-type')
    parser.add_argument("-n", "--node_name", required=True, help='node name: the name of node the pod runing with')
    parser.add_argument("-p", "--pod_name", required=True, help='pod name: the name of the running pod')
    parser.add_argument("-l", "--labels", required=True, help='label_config_file_env: where the labels.yaml should be stored')
    parser.add_argument("-c", "--auto_pipe_config", default=AUTO_PIPE_CONFIG, help='input config')
    parser.add_argument("-f", "--recon_fps", default=2, help='reconstruction fps')
    parser.add_argument("-o", "--overwrite_loc_fps_with_full_fps", action="store_true", help='overwirte loc fps in template with full fps of video')
    
    input_vals = parser.parse_args()
    video_env = input_vals.video
    movement_type_env = input_vals.movement_type
    tower_repo_env = input_vals.tower_repo
    my_node_name_env = input_vals.node_name
    my_pod_name_env = input_vals.pod_name
    labels_path_env = input_vals.labels
    auto_pipe_config_env = input_vals.auto_pipe_config
    
    recon_fps = input_vals.recon_fps
    overwrite_loc_fps_with_full_fps = input_vals.overwrite_loc_fps_with_full_fps
    
    video = decode_env(video_env)[0]
    movement_type = decode_env(movement_type_env)[0]
    tower_dir = decode_env(tower_repo_env)[0]
    my_node_name = decode_env(my_node_name_env)[0]
    my_pod_name = decode_env(my_pod_name_env)[0]
    auto_pipe_config_file = decode_env(auto_pipe_config_env)[0]

    labels_path = decode_env(labels_path_env)[0]
    tower_operations_path = labels_path.replace('labels.yml', 'tower_operations.yml')
    proj_name = tower_dir.split('/')[-1]
        
    tower_commit_ID = tower_dir.split("/")[-1] 
    logger_name = "{}_run_at_{}_{}.log".format(tower_commit_ID, my_node_name, my_pod_name)
    auto_pipe_config = safe_yaml_load(auto_pipe_config_file)
    logger_dir = auto_pipe_config['super_proj']['log_path']
    template_file = auto_pipe_config['super_proj']['template_file']
    proj_root = os.path.join(tower_dir, 'sfm_proj')
    output_camera_pose_path = os.path.join(tower_dir, CAM_POSE_NAME)

    with open(os.path.join(tower_dir, movement_type), 'r') as f:
        movement_dict = json.load(f)
    if movement_dict['movement_type'] == "stationary":
        video_length = VideoInfo(os.path.join(tower_dir, video)).get_duration()
        gen_dummy_campose(output_camera_pose_path, video_length*1000)
    else: 
        if overwrite_loc_fps_with_full_fps:
            info = VideoInfo(os.path.join(tower_dir, video))
            loc_fps = info.get_framerate()
        else:
            loc_fps = None
        super_src = SuperProjectSrcTower(tower_dir, \
                                    video, proj_root, \
                                    recon_fps=recon_fps, loc_fps=loc_fps)
        super_cfg_src_file = super_src.gen_super_src_yaml()
        SuperProject(super_cfg_src_file, template_file, logger_dir=logger_dir, logger_name = logger_name).run()
        export_sfm_trajectory(tower_dir, output_camera_pose_path, route_file='filled_routes.json')
    gen_tower_operations(tower_operations_path, tower_dir)
