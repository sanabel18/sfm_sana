import json
import yaml
import logging
import os
import shutil
import traceback

from preproc import PreprocessLocTask, PreprocessReconTask, AllSamplingTask, UniformVideoSamplingTask, UniformRigSamplingTask, PersonMaskTask, SlicingTask, InitReconLocTask, SlicingTask_w_keyIdx
from sfm import InitializeSfM, ComputeFeatures, VisualizeFeatures, GeneratePairList, ComputeMatches, VisualizeMatches, IncreRecon, IncreReconV2, RefineSfMRCampose, IncreReconRefine, FilterSfMResults, ExportSfMResults, ExpandFPS, MatchSub2Main
from mvs import DensifyPCL, GenPoissonMesh, ReconMesh, RefineMesh, TextureMesh
from postproc import CleanMesh, SimplifyMesh, CorrectLevel, GenFootprints, GenFootprintsByTrimesh, ZeroInitPose, ExportModel2App, CalcTransSub2Main, ApplyTrfExportSubs, Copy2Tower, ConcatConvert2Geojson, ConcatConvert2GeojsonGPSAnchored, GenStepSizeAnchor, FillRoute
from utils.genutil import safely_create_dir, get_current_time_str
from utils.logger import get_logger, close_all_handlers
from utils.safe_yaml import safe_yaml_load

from os.path import join


task_classes = {
    'PreprocessLocTask': PreprocessLocTask,
    'PreprocessReconTask': PreprocessReconTask,
    'AllSamplingTask': AllSamplingTask,
    'UniformVideoSamplingTask': UniformVideoSamplingTask,
    'UniformRigSamplingTask': UniformRigSamplingTask,
    'PersonMaskTask': PersonMaskTask,
    'SlicingTask': SlicingTask,
    'SlicingTask_w_keyIdx': SlicingTask_w_keyIdx,
    'InitReconLocTask': InitReconLocTask,
    'SfM_INITIALIZE': InitializeSfM,
    'SfM_COMPUTE_FEATURES': ComputeFeatures,
    'SfM_VISUALIZE_FEATURES': VisualizeFeatures,
    'GENERATE_PAIR_LIST': GeneratePairList,
    'SfM_COMPUTE_MATCHES': ComputeMatches,
    'SfM_VISUALIZE_MATCHES': VisualizeMatches,
    'SfM_INCREMENTALLY_RECONSTRUCT': IncreRecon,
    'SfM_INCREMENTALLY_RECONSTRUCT_V2': IncreReconV2,
    'SfM_REFINE_CAMPOSE': RefineSfMRCampose,
    'SfM_INCREMENTALLY_RECONSTRUCT_REFINE': IncreReconRefine,
    'SfM_FILTER_RESULTS': FilterSfMResults,
    'SfM_EXPORT_RESULTS': ExportSfMResults,
    'SfM_EXPAND_FPS': ExpandFPS,
    'MVS_DENSIFY_POINT_CLOUD': DensifyPCL,
    'MVS_GENERATE_POISSON_MESH': GenPoissonMesh,
    'MVS_RECONSTRUCT_MESH': ReconMesh,
    'MVS_REFINE_MESH': RefineMesh,
    'MVS_TEXTURE_MESH': TextureMesh,
    'CLEAN_MESH': CleanMesh,
    'SIMPLIFY_MESH': SimplifyMesh,
    'CORRECT_LEVEL': CorrectLevel,
    'GENERATE_FOOTPRINTS': GenFootprints,
    'GENERATE_STEPSIZE_ANCHOR': GenStepSizeAnchor,
    'GENERATE_FOOTPRINTS_BY_TRIMESH': GenFootprintsByTrimesh,
    'ZERO_INITIAL_POSE': ZeroInitPose,
    'EXPORT_MODEL_TO_APP': ExportModel2App,
    'FILL_ROUTE': FillRoute,
    'SfM_MATCH_SUB2MAIN': MatchSub2Main,
    'CALCULATE_TRANSFORMATION_SUB2MAIN': CalcTransSub2Main,
    'APPLY_TRANSFORMATION_EXPORT_SUBS': ApplyTrfExportSubs,
    'COPY_TO_TOWER': Copy2Tower,
    'CONCAT_CONVERT_TO_GEOJSON': ConcatConvert2Geojson,
    'CONCAT_CONVERT_TO_GEOJSON_GPS_ANCHORED': ConcatConvert2GeojsonGPSAnchored,
}


class Project(object):
    def __init__(self, cfg: dict):
        '''
        Setup the project object, i.e. create:
            1. logger
            2. done tasks dict
            3. task objects
        '''
        
        # 1. Setup project logger
        self.logger = get_logger('Project.log', self.dirs['log'])        
        
        # 2. Initialize done tasks dict
        self.done_tasks_path = join(self.dirs['prj'], 'done_tasks.json')
        try:
            with open(self.done_tasks_path, 'r') as f:
                self.done_tasks = json.load(f)  # Load done tasks file
        except:  # Either file DNE or is empty
            self.done_tasks = {}
        
        # 3. Create all tasks listed in pipeline
        self.task_dict = {}
        self.pipeline = self.cfg['project']['pipeline']
        for step, task_name in enumerate(self.pipeline):
            if task_name in self.done_tasks:
                self.logger.warn(f'Task {task_name} was already done before, ignore it.')
            else:
                self.create_task(step, task_name)

        # List all members of the object in log
        self.logger.info(f'Object: {self.__dict__}')        

        # After initialization successfully finished, copy the config to the log directory
        with open(join(self.dirs['log'], 'record.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return
    
    def create_dynamic_folders(self):
        ''' Content should be defined in child classes. '''
        pass

    def create_static_folders(self, folder_struct: dict):
        ''' Create all static directories (whose names stay the same for all project instances). '''
        
        for parent_dir in folder_struct:
            for sub_dir in folder_struct[parent_dir]:
                self.dirs[sub_dir] = safely_create_dir(self.dirs[parent_dir], sub_dir)        
        return
    
    def create_task(self, step: int, task_name: str):
        ''' Create task objects with loggers. '''
        
        self.task_dict[task_name] = task_classes[task_name](step, self.dirs, self.cfg)
        return
    
    def run_pipeline(self):
        ''' Run all tasks listed in pipeline, and kill the logger properly. '''

        returncode = 0 
        try:
            for step, task_name in enumerate(self.pipeline):
                # Ignore the step if it's already done before
                if task_name in self.done_tasks:
                    continue
                
                # Run the task
                result = self.task_dict[task_name].run()
                returncode = result.returncode

                # Update done task json with timestamp if task is successfully done
                if returncode == 0:
                    self.done_tasks[task_name] = get_current_time_str()
                    self.logger.info(f'Step {step:02d}: {task_name} successfully done at {self.done_tasks[task_name]}.')
                    with open(self.done_tasks_path, 'w') as f:  # Yes the existing file should be overwritten
                        json.dump(self.done_tasks, f)
                else:
                    self.logger.info(f'Step {step:02d}: {task_name} failed with returncode {returncode}.')
                    return returncode
        
        except Exception as e:
            e = traceback.format_exc()
            self.logger.error(f'Pipeline failed with message {e}.')

        finally:
            close_all_handlers(self.logger)
        
        return returncode


class PreprocProj(Project):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Make directories
        self.dirs = {}
        self.create_dynamic_folders()
        static_dirs = {
            'prj': ['img', 'img_upsmpl']
        }
        self.create_static_folders(static_dirs)
        
        # Other simple things
        super().__init__(cfg)
        return
    
    def create_dynamic_folders(self):
        ''' Create all dynamic directories (whose names vary for all project instances). '''
        
        time_str = get_current_time_str()
        
        # 1. Make project directory
        self.dirs['prj'] = self.cfg['project']['project_dir']
        if not os.path.exists(self.dirs['prj']): os.makedirs(self.dirs['prj'])
        
        # 2. Check folder structure in project directory & make data directory
        if self.cfg['project']['type'] == 'PreprocProj':
            pass
        else:
            raise RuntimeError('Invalid project type in config file.')
            
        # 3. Make log directory with datetime
        self.dirs['log'] = safely_create_dir(self.dirs['prj'], 'log_' + time_str)
        return

class ExportProj(Project):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        # Make directories
        self.dirs = {}
        self.create_dynamic_folders()
 
        # Other simple things
        super().__init__(cfg)
        return
     
    def create_dynamic_folders(self):
        ''' Create all dynamic directories (whose names vary for all project instances). '''
        
        time_str = get_current_time_str()
        
        # 1. Make project directory
        self.dirs['prj'] = self.cfg['project']['project_dir']
        if not os.path.exists(self.dirs['prj']): os.makedirs(self.dirs['prj'])
        
        # 2. Check folder structure in project directory & make data directory
        print("self.cfg {}".format(self.cfg))
        if self.cfg['project']['type'] == 'ExportProj':
            pass
        else:
            raise RuntimeError('Invalid project type in config file.')
            
        # 3. Make log directory with datetime
        self.dirs['log'] = safely_create_dir(self.dirs['prj'], 'log_' + time_str)
        return


            
class ReconProj(Project):
    def __init__(self, cfg: dict):
        self.cfg = cfg

        # Make directories
        self.dirs = {}
        self.create_dynamic_folders()
        static_dirs = {
            'prj': ['subsidiaries'],
            'data': ['matches'],
            'log': [],
            'subsidiaries': ['sfm', 'loc', 'mvs', 'postproc'],
            'sfm': ['undistorted'],
            'loc': ['loc_matches'],
            'postproc': ['seg', 'tmp']
        }
        self.create_static_folders(static_dirs)

        # Symbolic link to image project
        self.dirs['img_prj'] = join(self.dirs['data'], 'img_prj')
        try:
            os.symlink(self.cfg['images']['img_prj_dir'], self.dirs['img_prj'])
        except FileExistsError as e:
            pass  # Simply trust that the existing thing is right
        self.dirs['img'] = join(self.dirs['img_prj'], 'img')
        self.dirs['img_upsmpl'] = join(self.dirs['img_prj'], 'img_upsmpl')
        
        # Other simple things
        super().__init__(cfg)
        return
    
    def create_dynamic_folders(self):
        ''' Create all dynamic directories (whose names vary for all project instances). '''
        
        time_str = get_current_time_str()
        
        # 1. Make project directory
        self.dirs['prj'] = self.cfg['project']['project_dir']
        if not os.path.exists(self.dirs['prj']): os.makedirs(self.dirs['prj'])
        
        # 2. Check folder structure in project directory & make data directory
        if self.cfg['project']['type'] == 'ReconProj':
            self.dirs['data'] = safely_create_dir(self.dirs['prj'], 'data_' + self.cfg['project']['name'])
        else:
            raise RuntimeError('Invalid project type in config file.')
            
        # 3. Make log directory with datetime
        self.dirs['log'] = safely_create_dir(self.dirs['prj'], 'log_' + time_str)
        return
                

class LocProj(Project):
    def __init__(self, cfg: dict):
        

        self.cfg = cfg

        # Check the project definition & identify the type of source projects
        self.dirs = {}
        self.dirs['prj'] = self.cfg['project']['project_dir']
        #self.dirs['src_prj'] = self.cfg['project']['src_dir']
        if ('tower_src_dir' in self.cfg['project']):
            self.dirs['src_prj'] = self.cfg['project']['tower_src_dir']
        if not os.path.exists(self.dirs['prj']): os.makedirs(self.dirs['prj'])
        self.dirs['main_prj'] = self.cfg['transform']['prj_dir_main']
        self.dirs['sub_prj'] = self.cfg['transform']['prj_dir_sub']
        self.cfg['transform']['type_main_prj'] = self.identify_source_prj_type(
            self.dirs['main_prj'], self.cfg['transform']['data_dir_name_main'])
        self.cfg['transform']['type_sub_prj'] = self.identify_source_prj_type(
            self.dirs['sub_prj'], self.cfg['transform']['data_dir_name_sub'])
        
        self.validate_prj_def()
        
        # Paths of important directories of main and sub project
        self.dirs['main_data'] = join(self.dirs['main_prj'], self.cfg['transform']['data_dir_name_main'])
        self.dirs['main_matches'] = join(self.dirs['main_data'], 'matches')
        self.dirs['sub_data'] = join(self.cfg['transform']['prj_dir_sub'], self.cfg['transform']['data_dir_name_sub'])
        self.dirs['sub_img'] = join(join(self.dirs['sub_data'], 'img_prj'), 'img')

        # Make directories except for project folder
        self.create_dynamic_folders()
        static_dirs = {
            'loc': ['matches', 'query']
        }
        self.create_static_folders(static_dirs)

        # Other simple things
        super().__init__(cfg)
        return

    def create_dynamic_folders(self):
        ''' Create all dynamic directories (whose names vary for all project instances). '''
        
        time_str = get_current_time_str()
        
        # 2. Check folder structure in project directory & make loc__[sub]__2__[main] directory
        if self.cfg['project']['type'] == 'LocProj':
            self.dirs['loc'] = safely_create_dir(
                self.dirs['prj'], 'loc__' + self.cfg['transform']['data_dir_name_sub'][5:] + '__2__' + self.cfg['transform']['data_dir_name_main'][5:])
        else:
            raise RuntimeError('Invalid project type in config file.')
            
        # 3. Make log directory with datetime
        self.dirs['log'] = safely_create_dir(self.dirs['prj'], 'log_' + time_str)
        return
    
    def identify_source_prj_type(self, source_prj_path: str, data_dir_name: str):
        ''' Identify if the source project is a localization project or reconstruction project. '''
        
        n_data_dir = 0
        data_dir_exists = False
        for item_name in os.listdir(source_prj_path):
            if item_name[0:5] == 'data_': n_data_dir += 1
            if item_name == data_dir_name: data_dir_exists = True
        if data_dir_exists and (n_data_dir > 1):
            type_source_prj = 'LocProj'
        elif data_dir_exists and (n_data_dir == 1):
            type_source_prj = 'ReconProj'
        else:
            err = 'Error reason: \n'
            if not data_dir_exists: err = f'The specified source project {data_dir_name} does not exist. \n'
            if n_data_dir < 1: err += f'There is no data directory in source project folder {source_prj_path}.'
            raise RuntimeError(err)
        return type_source_prj

    def validate_prj_def(self) -> bool:
        '''
        If the running project folder already exists, then it must be the main project and this main project must be a localization project, in order to avoid mess.
        If creating a new project folder, then the main project must be a reconstruction project, in order to avoid redundant projects.
        '''
                
        error_msg = ''
        is_resuming_prj = self.if_resuming_prj()
        #self.logger.info(f'is resuming_prj: {is_resuming_prj}')
        if os.listdir(self.dirs['prj']) and (not is_resuming_prj):  # If there is something inside the project folder then that means it's already an existing project
            if (self.dirs['prj'] != self.dirs['main_prj']):
                error_msg += 'It is not allowed to use an arbitrary existing folder other than main project folder as running localization project folder. \n'
            if self.cfg['transform']['type_main_prj'] != 'LocProj':
                error_msg += 'It is not allowed to use an existing folder whose type is not localization as running localization project folder. \n'
        else:            
            if ((self.cfg['transform']['type_main_prj'] != 'ReconProj') and
                (self.cfg['transform']['prj_dir_main'] != self.cfg['project']['project_dir']) and
                (not self.cfg['project']['force_new_project_dir'])):
                error_msg += 'It is not allowed to make a new project folder if the main project is already a localization project, unless you force it explicitly. \n'            
        
        if error_msg != '':
            raise RuntimeError(error_msg)
        
        if (not is_resuming_prj) and (self.cfg['transform']['type_main_prj'] == 'LocProj'):
            print('You want to localize to a LocProj, so the done_tasks.json will be deleted.')
            try:
                os.remove(join(self.dirs['main_prj'], 'done_tasks.json'))
            except:
                print(f'Failed to remove done_tasks.json...')
        return True
    
   
    def if_resuming_prj(self):
        sorted_log_dirs = sorted([item for item in os.listdir(self.dirs['prj']) if item[0:4]=='log_'])
        if_same_config = False
        for log_dir in sorted_log_dirs:
            if os.path.exists(join(join(self.dirs['prj'], log_dir), 'record.yaml')):
                existing_cfg = safe_yaml_load(join(join(self.dirs['prj'], log_dir), 'record.yaml'))
                if_same_config_tmp = self.whether_same_configs(self.cfg, existing_cfg) 
                if_same_config = if_same_config or if_same_config_tmp
        return if_same_config

    def whether_same_configs(self, cfg1: dict, cfg2: dict):
        ''' Compare the important variables in config, in order to determine if they should be regarded as the same project. '''
        
        # Preparation
        is_same = True
        comparison_list = [['project', 'project_dir'],
                           ['transform', 'prj_dir_main'], ['transform', 'data_dir_name_main'],
                           ['transform', 'prj_dir_sub'], ['transform', 'data_dir_name_sub']]
        
        # As long as the two configs have same value for all entries in the comparison list, treat them as the same
        for keylist in comparison_list:
            #self.logger.info(f'cfg1, cfg2 {cfg1[keylist[0]][keylist[1]]} {cfg2[keylist[0]][keylist[1]]}')
            if cfg1[keylist[0]][keylist[1]] != cfg2[keylist[0]][keylist[1]]:
                is_same = False
                return is_same
        return is_same
