import os
import os.path as osp
import glob
import json
import shutil

from pytower.tower_agent import TowerAgent
from pytower.tower_exception import AddError, CleanError, DownloadError, SetupError
from pytower.tower_utils import download_file
from tool.tower_stage_manager.tower_file_genutils import gen_repo_basename, \
        gen_folder_path_from_base
import logging

class AddProcess:
    def __init__(self):
        pass
    def run(self, tower_agent):
        try:
            tower_agent.add()
            return True, str()
        except Exception as e:
            repo_name = tower_agent.get_repo_name()
            err = f'repo name: {repo_name} add files to tower failed.'
        return False, err

class LabelProcess:
    """
    add label to file
    Args:
    label_param: LabelParam
    """
    def __init__(self, label_param):
        self.label_param = label_param
        self.label_to_file = self.label_param.get_label_to_file_dict()
    def run(self, tower_agent):
        repo_path = tower_agent.get_repo_path()
        try:
            for label in self.label_to_file.keys():
                file_list = self.label_to_file[label]
                for f in file_list:
                    file_path = os.path.join(repo_path,f)
                    file_path = [file_path]
                    tower_agent.label_files(file_path, label, ignore_non_existed_error=False)
            return True, str()
        except Exception as e:
            repo_name = tower_agent.get_repo_name()
            err = f'repo name: {repo_name} label files failed'
            return False, err

class CommitProcess:
    """
    Args:
    commit_param: CommitParam
    """
    def __init__(self, commit_param):
        self.commit_param = commit_param
        self.done = None
    def run(self, tower_agent):
        commit_msg = self.commit_param.get_commit_msg()
        try:
            tower_agent.commit(commit_msg=commit_msg)
            return True, str()
        except Exception as e:
            repo_name = tower_agent.get_repo_name()
            err = f'repo name: {repo_name} commit failed.'
            return False, err

class AddTagProcess:
    """
    Args:
    add_tag_param: AddTagParam
    """
    def __init__(self, add_tag_param):
        self.add_tag_param = add_tag_param
        self.done = None
    def run(self, tower_agent):
        tag = self.add_tag_param.get_dst_commit_tag()
        try:
            tower_agent.tag(tag)
            return True, str()
        except:
            repo_name = tower_agent.get_repo_name() 
            err = f'repo name: {repo_name} add tag failed'
            return False, err

class CleanUpprocess:
    def __init__(self):
        pass

    def rm_downloaded_files(self, repo_path):
        clone_repo_folder = os.path.basename(repo_path)
        repo_path_base = os.path.dirname(repo_path)
        repo_name_base = clone_repo_folder.split('tag')[0]

        dir_list = glob.glob(os.path.join(repo_path_base, repo_name_base+'*'))
        for dirname in dir_list:
            shutil.rmtree(dirname)

    def run(self,tower_agent):
        """
        remove related file folders
        1. cloned repo
        2. downloaded files
        Args:
        tower_agent: TowerAgent
        """
        repo_path = tower_agent.get_repo_path()
        try:
            tower_agent.cleanup()
            self.rm_downloaded_files(repo_path)
            return True, str()
        except Exception as e:
            repo_name = tower_agent.get_repo_name()
            err = f'repo name: {repo_name} clean up failed'
            return False, err

class TowerFilePostProcessor:
    """
    post process
    Args:
    label_param: LabelParam
    commit_param: CommitParam
    add_tag_param: AddTagParam
    ws_root: str: path of worksapce
    commit_tag: str: tag of main parent
    """
    def __init__(self, label_param, commit_param, add_tag_param, ws_root, commit_tag):
        self.commit_tag = commit_tag
        self.ws_root = ws_root
        self.label_param = label_param
        self.commit_param = commit_param
        self.add_tag_param = add_tag_param
        self.add_process = self._create_add_process()
        self.label_process = self._create_label_process()
        self.commit_process = self._create_commit_process()
        self.add_tag_process = self._create_add_tag_process()
        self.clean_up_process = self._create_clean_up_process()

    def _create_add_process(self):
        return AddProcess()
    
    def _create_label_process(self):
        return LabelProcess(self.label_param)

    def _create_commit_process(self):
        return CommitProcess(self.commit_param)

    def _create_add_tag_process(self):
        return AddTagProcess(self.add_tag_param)

    def _create_clean_up_process(self):
        return CleanUpprocess()
   
    def run(self, repo_name, has_been_done):
        """
        if has_been_done is True, process will be ignored
        Go through processes:
        1. init tower obj
        2. add file
        3. label file
        4. commit tower repo
        5. tag commit
        
        result will be reported as  
        status: dict{key: repo_path, value:{keys: process_name, value: True/False}}
        
        Args:
        repo_path: str, local path of tower repo
        has_been_done: bool
        Return:
        bool: True if there is no fail on 1~5
        status: dict
        """
        repo_name_base = gen_repo_basename(repo_name)
        tag = f'tag_{self.commit_tag}'
        repo_path = gen_folder_path_from_base(self.ws_root, repo_name_base, tag)
  
        if has_been_done:
            return True, str()
        else:
            try:
                tower_agent = TowerAgent(repo_path, clone=False)
            except Exception as e:
                status = f'repo name: {repo_name} tower init failed'
                return False, status

            add_success, status = self.add_process.run(tower_agent)
            if not add_success:
                return False, status
            
            label_success, status = self.label_process.run(tower_agent)
            if not label_success:
                return False, status

            if tower_agent.check_need_to_commit():
                commit_success, status  = self.commit_process.run(tower_agent)
                if not commit_success:
                    return False, status

            add_tag_success, status = self.add_tag_process.run(tower_agent)
            if not add_tag_success:
                return False, status
            
            clean_up_success, status = self.clean_up_process.run(tower_agent)
            if not clean_up_success:
                return False, status

            return True, status


