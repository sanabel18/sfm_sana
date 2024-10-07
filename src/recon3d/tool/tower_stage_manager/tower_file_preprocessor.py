import os
import os.path as osp
import glob
import json
import shutil
import copy

from pytower.tower_agent import TowerAgent
from pytower.tower_exception import CloneError, CleanError, DownloadError, SetupError
from pytower.tower_utils import download_file
from tool.tower_stage_manager.tower_file_genutils import gen_repo_basename, \
        gen_folder_path_from_base
import logging

class DownloadProcess:
    """
    1. download file list defined within download_param
    2. check if the number of the download files same as defined
       in download_param
    Args: 
    download_param: DownloadParam
    ws_root: str: workspace where you want to download the stage
    commit_tag: str: the commit tag of cloned repo
    """
    def __init__(self, download_param, ws_root, commit_tag):
        self.commit_tag = commit_tag
        self.work_dir = ws_root
        self.input_data = download_param.get_input_data()
    
    def download_files(self, repo_name, output_data):
        """
        output_data has same data struct defined in DownloadParam.input_data
        but with file-number replaced with list of file_abs_path
        key: commit-tag:string, value:dict{key:file-label, value: ['/aaa/bb/file1','/aaa/bb/file2']}
 
        go through tag in output_data, and download files with different file labels
        and stores in folder with naming convention: 
        'repo_name_base_{tag}_{file_label}'

        check files downloaded and update the file path list of the downloaded files into output_data

        Args:
        repo_name: str
        output_data: dict: DownloadParam.input_data
        """
        for commit_tag in output_data.keys():
            if commit_tag == self.commit_tag:
                continue
            for file_label in output_data[commit_tag].keys():
                repo_name_base = gen_repo_basename(repo_name)
                save_path = gen_folder_path_from_base(self.work_dir, repo_name_base, f'tag_{commit_tag}_label_{file_label}')
                file_label_list = self._format_file_label(file_label)
                download_file(save_path, file_label_list, repo_name, commit_tag)
                downloaded_file_list = self._list_downloaded_files(save_path) 
                output_data[commit_tag][file_label] = downloaded_file_list
    
    def _list_downloaded_files(self, file_path):
        """
        list files in file_path and export it absolute path
        as list of string
        
        Args:
        file_path: str
        Return: list of str
        """
        file_path = os.path.abspath(file_path)
        return glob.glob(f'{file_path}/*')
        
 
    def _format_file_label(self, file_label):
        """
        Args: 
        file_label: str
        Return:
        list of str
        """
        file_label = '@'+file_label
        return [file_label]
 
    def compare_output_dict_with_input_dict(self, output_data):
        """
        check if the length of file_path_list in output_data is consistant with 
        file_number stored in self.input_data
        Args:
        output_data: dict: DownloadParam.input_data
        Returns:
        bool
        err: str
        """
        for tag in self.input_data.keys():
            flabel_to_fnum = self.input_data[tag]
            for label in flabel_to_fnum.keys():
                file_number = flabel_to_fnum[label]
                file_list_out = output_data[tag][label]
                if file_list_out:
                    file_number_out = len(file_list_out)
                    if file_number != file_number_out:
                        err = f'number of file with tag: {tag}, label: {label} should be {file_number}, not {file_number_out}.'
                        return False, err
                else:
                    err = f'file_list of tag: {tag}, label: {label} do not exist.'
                    return False, err
        return True, str()

    def update_output_data_cloned(self, tower_agent, output_data):
        """
        check files cloned in main repo 
        check file exists and recorded file path
        
        output_data has same data struct defined in DownloadParam.input_data
        but with file-number replaced with list of file_abs_path
        key: commit-tag:string, value:dict{key:file-label, value: ['/aaa/bb/file1','/aaa/bb/file2']}
 
        Args:
        tower_agnet: TowerAgent
        output_data: dict: DownloadParam.input_data
        """
        for file_label in output_data[self.commit_tag].keys():
            file_list = tower_agent.get_file_by_label(file_label)
            if self.check_files_exist(file_list):
                output_data[self.commit_tag][file_label] = file_list
    

    def check_files_exist(self, file_list):
        """
        check all files in file_list exists

        Args:
        file_list: list of str
        Retrun:
        bool
        """
        for f in file_list:
            exist = os.path.exists(f)
            if not exist:
                return False
        return True

    def run(self, tower_agent, output_data):
        """
        we will clone tower repo from main parent
        and files downloaded from side parents

        the folder naming conventation:
        main parent:
        repo_name_base + '_tag_{commit_tag}'
        side parent:
        repo_name_base + '_tag_{commit_tag}_label_{file_label}'
        
        example:
        main repo:     _大甲媽場域_20210409_112930_961113763_tag_marker
        side parents:  _大甲媽場域_20210409_112930_961113763_tag_rtk_gps_cut_edit_label_gpstrack
                       _大甲媽場域_20210409_112930_961113763_tag_rtk_gps_cut_edit_label_video
        
        
        output_data has same data struct defined in DownloadParam.input_data
        but with file-number replaced with list of file_abs_path
        key: commit-tag:string, value:dict{key:file-label, value: ['/aaa/bb/file1','/aaa/bb/file2']}
 
        Args: 
        tower_agent: TowerAgent
        output_data: dict: DowanloadParam.input_data
        Return:
        bool: Ture if all files are checked from main repo and side parent
        str: error message , empty string if no error occured
        """
 
        try:
            repo_name = tower_agent.get_repo_name()
            self.download_files(repo_name, output_data)
            self.update_output_data_cloned(tower_agent, output_data) 
            if_download_n_cloned, err  = self.compare_output_dict_with_input_dict(output_data)
            return if_download_n_cloned, err
        except Exception as e:
            err = f'repo: {repo_name} with error {e} during clone/download process'
            return False, err

        

class DataValidationProcess:
    """
    Validate the content of downloaded files with user defined function
    the structure of data_validation dict looks like:
    key: commit_tag:string, value: dict{key:file-label, value:func-pointer}
    example: 
    data_validation_dict = \
        {'marker':{'marker':check_src_marker}, 'release':{'marker':check_src_marker}}
    
    Args:
    data_validation_dict
    """
    def __init__(self, data_validation_dict):
        self.data_validation_dict = data_validation_dict
    
    def run(self, output_data):
        for tag in self.data_validation_dict.keys():
            file_dict = self.data_validation_dict[tag]
            for file_label in file_dict.keys():
                file_path_list = output_data[tag][file_label]
                validate_func = file_dict[file_label]
                for file_path in file_path_list:
                    validated = self.validate(file_path, validate_func)
                    if not validated:
                        err = f'non-validated file: {file_path}'
                        return False, err
        return True, str()
   
    def validate(self,filename, func):
        """
        validate content of filename with function pointer func
        Args: 
        filename: str
        func: function pointer
        Returns:
        bool, if validated
        """
        validated = func(filename)
        return validated
        
class TowerFilePreProcessor:
    """
        Handle file-related tasks
        1. download files
        2. validate download data
    data_validation_dict defines whilch files should be validated with 
    user-defined function
    key: commit_tag:string, value: dict{key:file-label, value: user-defined func}
    example:
    data_validation_dict = \
            {'node-marker':{'marker':check_src_marker},
             'sfm-gps':{'sfm-gps-rtk':check_geojson}}
    check_src_marker and check_geojson should be defined by user

    Args:
    download_param: DownloadParam
    data_validation_dict: dict
    """

    def __init__(self, download_param, data_validation_dict, ws_root, commit_tag):
        self.download_param = download_param
        self.input_data = download_param.get_input_data()
        self.commit_tag = commit_tag
        self.ws_root = ws_root
        self.data_validation_dict = data_validation_dict
        self.download_process = self._create_download_process()
        self.data_validation_process = self._create_data_validation_process()
    
    def _create_download_process(self):
        return DownloadProcess(self.download_param, self.ws_root, self.commit_tag)
    
    def _create_data_validation_process(self):
        return DataValidationProcess(self.data_validation_dict)
    
    def _create_output_data(self):
        """
        create same dict structure as self.input_data
        and init all attribute to None
        """
        output_data = copy.deepcopy(self.input_data)
        for tag in output_data.keys():
            output_data_tag = output_data[tag]
            for flabel in output_data_tag.keys():
                output_data_tag[flabel] = None
        return output_data
 
    def run(self, repo_name, has_been_done):
        """
        create tower_agent and run 
        1. download_process
        2. data_validation_process
        
        result status is a string with error message 
 
        Args:
        repo_name: str
        has_been_done: bool 
        
        Return:
        bool: True if 1. and 2. are both True
        status: str
        """
        repo_name_base = gen_repo_basename(repo_name)
        tag = f'tag_{self.commit_tag}'
        main_parent_path = gen_folder_path_from_base(self.ws_root, repo_name_base, tag)
        output_data = self._create_output_data() 

        try:
            tower_agent = TowerAgent(main_parent_path, clone=True, \
                                    identity=repo_name, clone_mode='tag',\
                                    query_tag=self.commit_tag)
        except Exception as e:
            status = f'repo name: {repo_name} init with errer {e}'
            return False, status


        download_success, status  = self.download_process.run(tower_agent, output_data)
        if not download_success:
            return False, status

        data_validation_success, status  = self.data_validation_process.run(output_data) 
        if not data_validation_success:
            return False, status
        
        return True, status
