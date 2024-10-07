from tool.tower_stage_manager.tower_file_postprocessor import TowerFilePostProcessor
from utils.safe_yaml import safe_yaml_load
from tool.tower_stage_manager.input_params import DownloadParam, CommitParam, LabelParam, AddTagParam
from tool.tower_stage_manager.tower_file_preprocessor import TowerFilePreProcessor
from tool.tower_stage_manager.validation_func import check_src_marker, VALIDATION_DICT
from tool.tower_stage_manager.tower_stage_manager import TowerStageManager
import pickle, os, argparse


def run():
    """
    script launching tower_stage_manager

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help='input config')
    parser.add_argument("-p", "--process_type", required=True, help='process_type: preprocess or postprocess')
    parser.add_argument("-i", "--use_stage_id", action="store_true")
    input_vals = parser.parse_args()
    input_config_file = input_vals.config
    process_type = input_vals.process_type
    use_stage_id = input_vals.use_stage_id
    input_config = safe_yaml_load(input_config_file)
    
    if process_type == 'preprocess':
        input_data = input_config['input_data']
        if use_stage_id:
            stage_id = input_config['stage_id']
            repo_name_list = []
        else:
            if 'repo_name_list_file' in input_config:
                repo_name_list_file = input_config['repo_name_list_file']
                repo_name_list = pickle.load(open(repo_name_list_file, 'rb'))
                stage_id = str()
            else:
                e = f'repo_name_list_file not defined in config'
                raise ValueError(e)
 
        ws_root = input_config['ws_root']
        repo_tag = input_config['repo_tag']
        status_root_path = input_config['status_root_path_preprocess']
        label_to_tag = input_config['data_validation_label_to_tag']
        data_validation_dict = gen_data_validation_dict(label_to_tag)
        download_param = DownloadParam(input_data)
        process = TowerFilePreProcessor(download_param, data_validation_dict, ws_root, repo_tag)

    elif process_type == 'postprocess':
        commit_msg = input_config['commit_msg']
        commit_tag = input_config['commit_tag']
        label_to_file = input_config['label_to_file']
         
        repo_name_src_path  = input_config['repo_name_src_path']
        status_root_path = input_config['status_root_path_postprocess']
        ws_root = input_config['ws_root']
        repo_tag = input_config['repo_tag']
        
        commit_param = CommitParam(commit_msg)
        label_param = LabelParam(label_to_file)
        add_tag_param = AddTagParam(commit_tag)
        
        process = TowerFilePostProcessor(label_param, commit_param, add_tag_param, ws_root, repo_tag)
        
        complete_list_file = os.path.join(repo_name_src_path, 'complete_list.p')
        repo_name_list = pickle.load(open(complete_list_file, 'rb'))
        stage_id = str()
 
    else:
        e = 'process type {} is not supported'.format(process_type)
        raise ValueError(e)
    
    tower_stage_manager = TowerStageManager(process, status_root_path, stage_id = stage_id, repo_name_list = repo_name_list)
    tower_stage_manager.run_stage_repos()

def gen_data_validation_dict(label_to_tag):
    """
    convert label_to_tag to data_validation_dict
    
    the structure of data_validation dict looks like:
    key: commit_tag:string, value: dict{key:file-label, value:func-pointer}
    example: 
    data_validation_dict = \
        {'marker':{'marker':check_src_marker}, 'release':{'marker':check_src_marker}}
    """
    data_validation_dict = {}
    for label in label_to_tag.keys():
        tags = label_to_tag[label]
        for tag in tags:
            if not (tag in data_validation_dict.keys()):
                data_validation_dict[tag] = {} 
            if label in VALIDATION_DICT:
                data_validation_dict[tag][label] = VALIDATION_DICT[label]
            else:
                e = f'file label {label} does not have validation function implemented.'
                raise ValueError(e)
    return data_validation_dict

if __name__ == "__main__":
    run()
