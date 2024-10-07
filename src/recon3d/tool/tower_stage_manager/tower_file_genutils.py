import os
import json

def gen_repo_basename(repo_name):
    return repo_name.replace('/','_')
    
def gen_folder_path_from_base(work_dir, repo_name_base, tag_label):
    folder_name = repo_name_base + f'_{tag_label}'
    return os.path.join(work_dir, folder_name)

def get_tower_repo_path_from_repo_name(tower_stage_dir, repo_name_list, tag):
    repo_path_list = []
    for repo_name in repo_name_list:
        repo_name_base = gen_repo_basename(repo_name)
        repo_path = gen_folder_path_from_base(tower_stage_dir, repo_name_base, tag)
        repo_path_list.append(repo_path)
    return repo_path_list

def load_repo_name_list(repo_name_file):
    import pickle
    repo_name_list = pickle.load(open(repo_name_file, 'rb'))
    repo_basename_list = []
    for repo_name in repo_name_list:
        repo_basename = gen_repo_basename(repo_name)
        repo_basename_list.append(repo_basename)
    return repo_basename_list

def load_src_marker_from_path_list(src_marker_path_list):
    src_marker_data_list = []
    for src_marker_path in src_marker_path_list:
        src_file = os.path.join(src_marker_path, 'source_marker.json')
        data = json.load(open(src_file,'r'))
        src_marker_data_list.append(data)
    return src_marker_data_list
 
def load_src_marker_from_tower_repo(repo_path_list):
    '''
    from tower repo read source markers with label @marker
    '''
    from pytower.tower_agent import TowerAgent
    src_marker_data_list = []
    for repo_path in repo_path_list:
        tower_agent = TowerAgent(repo_path, clone=False)
        src_marker_file = tower_agent.get_file_by_label('marker')
        if (len(src_marker_file) > 1):
            e = f'more than one source marker file'
            raise RuntimeError(e)
        elif ( len(src_marker_file) < 1):
            e = f'no source marker file found'
            raise RuntimeError(e)
        src_marker = src_marker_file[0]
        with open(src_marker, 'r') as f:
            src_marker_data = json.load(f)
            src_marker_data_list.append(src_marker_data)
    return src_marker_data_list


