import yaml
import os
import pickle
from tower import Tower
import logging
from tool.tower_stage_manager.tower_file_genutils import gen_repo_basename, \
        gen_folder_path_from_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TowerStageManager:
    """
    assign tasks to process through repo_name_list
    repo_name_list can be provided directed by user, 
    or use stage_id to querry from tower server
    repo_name_list is list of tower repo names

    Args:
    process: TowerFilePostProcessor or TowerFilePreProcessor
    status_root_path: str, folder path to store log
    stage_id: str
    repo_namge_list: list of str
    """
    def __init__(self,\
                 process,\
                 status_root_path,\
                 stage_id=None,\
                 repo_name_list=None):
        self.complete_list = []
        self.process = process 
        self.status_root_path = status_root_path
        if stage_id and not repo_name_list:
            self.repo_name_list = self.set_repo_name_list(stage_id)
        elif repo_name_list and not stage_id:
            self.repo_name_list = repo_name_list
        else:
            err = f'TowerStageManager init: can only set one of stage_id or repo_name_list'
            raise ValueError(err) 
        if not os.path.isdir(status_root_path):
            os.makedirs(status_root_path)
        self.complete_list_last = self._load_complete_list()

    def set_repo_name_list(self, stage_id):
        return Tower.list_stage_repos(stage_id, timeout=30.0, quiet=False)
    
    def _load_complete_list(self):
        """
        load 'complete_list.p' from self.status_root_path
        if file not exists (first run), return empty list
        Returns: list of str, list of repo names
        """
        complete_list_file = os.path.join(self.status_root_path, 'complete_list.p')
        if os.path.isfile(complete_list_file):
            complete_list = pickle.load(open(complete_list_file, 'rb'))
        else:
            complete_list = []
        return complete_list

    def _export_to_complete_list(self, complete_list):
        """
        write complete_list to complete_list.p
        Args: list of str, list of repo names
        """
        complete_list_file = os.path.join(self.status_root_path, 'complete_list.p')
        pickle.dump(complete_list, open(complete_list_file,'wb'))
    
   
    def _export_status(self, status_list):
        """
        write status_list to file
        Args: 
        status_list: list of string
        Return:
        file "status.log" written in self.status_root_path
        """
        status_file = os.path.join(self.status_root_path,'status.log')
        f = open(status_file,'w')
        for status in status_list:
            f.writelines(status+'\n')
        f.close()

    def run_stage_repos(self):
        if len(self.repo_name_list) == 0:
            logger.info(f'repo name list is empty')
        else:
            failed_list = []
            for task in self.repo_name_list:
                has_been_done = task in self.complete_list_last
                completed, status  = self.process.run(task, has_been_done)
                if completed:
                    self.complete_list.append(task)
                    self._export_to_complete_list(self.complete_list)
                else:
                    failed_list.append(status)
            self._export_status(failed_list)
            if len(failed_list) == 0:
                logger.info('all jobs are done')
