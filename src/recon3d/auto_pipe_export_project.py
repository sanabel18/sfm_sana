import os
from super_project import SuperProject, SuperProjectSrc
import yaml
from utils.safe_yaml import safe_yaml_load
from auto_pipe_super_project import decode_env
from export_project import ExportProjectSrc, ExportProject
import argparse 

AUTO_PIPE_CONFIG = '../config/auto_pipe_config_stepsize.yaml'

def set_labels(labels_yaml, gpstrack, gpstrack_label):
    '''
    set label/unlabel files for tower auto pipline
    write to a yaml file at directory labels_yaml
    Here we shall label geo_data_sfm.geojson as @gpstrack
    and unlabel file that originally marked as @gpstrack

    Args: 
    labels_yaml: str, path to write label yaml
    gpstrack: str, filename originally marked as @gpstrack
    '''
    gpstrack_label = '@'+gpstrack_label
    labels = {'file': [{'path': 'geo_data_sfm.geojson', 
                        'labels': [gpstrack_label]} 
                       ]}
    with open(labels_yaml, 'w') as f:
        yaml.dump(labels, f, default_flow_style=False)

def add_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--tower_repo", required=True, help='main tower repo directory')
    parser.add_argument("-r", "--gps_repo", required=True, help='tower repo directory for gpstrack')
    parser.add_argument("-g", "--gpstrack", required=True, help='gps track: name of geojson file with tower label @gpstrack')
    parser.add_argument("-m", "--marker", required=True, help='source marker: name of source_marker.json with tower label @marker')
    parser.add_argument("-n", "--node_name", required=True, help='node name: the name of node the pod runing with')
    parser.add_argument("-p", "--pod_name", required=True, help='pod name: the name of the running pod')
    parser.add_argument("-l", "--labels", required=True, help='label_config_file_env: where the labels.yaml should be stored')
    parser.add_argument("-c", "--auto_pipe_config", default=AUTO_PIPE_CONFIG, help='input config')
    parser.add_argument("-a", "--gpstrack_label", help=f'name of gpstrack label')
    parser.add_argument("-u", "--use_cutted_video", action="store_true")
    
    input_vals = parser.parse_args()
    return input_vals

if __name__ == "__main__":
    '''
    This script will be used to activate Export Project by 
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
    gpstrack: name of geojson file with tower label @gpstrack
    markder: name of source_marker.json with tower label @marker
    node_name: the name of node the pod runing with
    pod_name: the name of the running pod
    labels: where the labels.yaml should be stored 
            ex: /aa/bb/labels.yaml
    '''
    
    input_vals = add_parser()
    gpstrack_env = input_vals.gpstrack
    src_marker_env = input_vals.marker
    tower_repo_env = input_vals.tower_repo
    gps_repo_env = input_vals.gps_repo
    my_node_name_env = input_vals.node_name
    my_pod_name_env = input_vals.pod_name
    label_loc_env = input_vals.labels
    auto_pipe_config_env = input_vals.auto_pipe_config
    gpstrack_label_env = input_vals.gpstrack_label
    
    use_cutted_video = input_vals.use_cutted_video
    
    gpstrack = decode_env(gpstrack_env)[0]
    src_marker = decode_env(src_marker_env)[0]
    tower_dir = decode_env(tower_repo_env)[0]
    gps_tower_dir = decode_env(gps_repo_env)[0]
    my_node_name = decode_env(my_node_name_env)[0]
    my_pod_name = decode_env(my_pod_name_env)[0]
    label_loc = decode_env(label_loc_env)[0]
    auto_pipe_config_file = decode_env(auto_pipe_config_env)[0]
    gpstrack_label = decode_env(gpstrack_label_env)[0]
    
    set_labels(label_loc, gpstrack, gpstrack_label)
        
    tower_commit_ID = tower_dir.split("/")[-1] 
    logger_name = "{}_run_export_at_{}_{}.log".format(tower_commit_ID, my_node_name, my_pod_name)
    auto_pipe_config = safe_yaml_load(auto_pipe_config_file)
    logger_dir = auto_pipe_config['export_proj']['log_path']
    template_file = auto_pipe_config['export_proj']['template_file']
    proj_root = os.path.join(tower_dir, 'sfm_proj')

    export_cfg_src = ExportProjectSrc(tower_dir, gps_tower_dir, src_marker, gpstrack, proj_root, use_cutted_video).get_export_src_dict() 
    export_cfg_template = safe_yaml_load(template_file)
    ExportProject(export_cfg_src, export_cfg_template, logger_dir=logger_dir, logger_name=logger_name).run()
 
