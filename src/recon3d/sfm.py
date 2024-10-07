import os
import json
import numpy as np
import re
import subprocess
import shutil
import itertools
import time
from abc import ABC, abstractmethod

from os.path import join
from scipy.sparse import dok_matrix
from task import Task
from utils.genutil import run_log_cmd, TransData, run_cmd_w_retry, reorder_sfm_data, filter_sfm_data
from utils.exportutil import export_sfm_data_visualization
from utils.safe_json import safe_json_dump


class InitializeSfM(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.focal_length = 1 if self.camera_model == 7 else prj_cfg['sfm']['focal_length']
        
        self.img_dir = prj_dirs['img']
        self.matches_dir = prj_dirs['matches']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the image listing command. '''

        self.logger.info('SfM initialization started.')
        cmd = ['openMVG_main_SfMInit_ImageListing',
               '-i', self.img_dir,
               '-o', self.matches_dir,
               '-c', str(self.camera_model),
               '-f', str(self.focal_length)]        
        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM initialization ended with return code {proc.returncode}.')
        return proc.returncode
    
    
class ComputeFeatures(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.describer = prj_cfg['sfm']['feature_matching']['describer']
        self.describer_preset = prj_cfg['sfm']['feature_matching']['describer_preset']
        
        self.matches_dir = prj_dirs['matches']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the feature computation command. '''

        self.logger.info('SfM feature computation started.')
        cmd = ['openMVG_main_ComputeFeatures',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-o', self.matches_dir,
               '-f', str(0),
               '-m', self.describer,
               '-p', self.describer_preset,
               '-u', str(0),
               '-n', str(self.n_thread)]
        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM feature computation ended with return code {proc.returncode}.')
        return proc.returncode


class VisualizeFeatures(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
                
        self.matches_dir = prj_dirs['matches']
        self.output_dir = join(prj_dirs['sfm'], 'visualize_features')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the feature visualization command. '''

        self.logger.info('SfM feature visualization started.')
        cmd = ['openMVG_main_exportKeypoints',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-d', self.matches_dir,
               '-o', self.output_dir]
        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM feature visualization ended with return code {proc.returncode}.')
        return proc.returncode
    

from pair_list_generator.slice_rig_pair_list_generator import SliceRigPairListGenerator
from pair_list_generator.backbone_detail_pair_list_generator import BackboneDetailPairListGenerator
from pair_list_generator.window_pair_list_generator import WindowPairListGenerator

class GeneratePairList(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        generator_type = prj_cfg['sfm']['pair_list']['generator_type']
        generator_config = prj_cfg['sfm']['pair_list']['generator_config']
        if generator_type == 'slice_rig':
            self.generator = SliceRigPairListGenerator(generator_config)
        elif generator_type == 'backbone_detail':
            self.generator = BackboneDetailPairListGenerator(generator_config)
        elif generator_type == 'window':
            self.generator = WindowPairListGenerator(generator_config)
        else:
            raise NotImplementedError

        self.sfm_data_file = join(prj_dirs['matches'], 'sfm_data.json')
        self.pair_list = join(prj_dirs['sfm'], 'pair_list.txt')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
    
    def run_task_content(self) -> int:
        ''' Generate pair list and then run the feature matching command. '''
        
        self.logger.info('Generating pair list...')    

        id_view_list = self.generator.sfm_data_to_id_view_list(self.sfm_data_file)
        _pair_list = self.generator.gen_pair_list(id_view_list)
        
        with open(self.pair_list, 'w') as f:
            [f.write('{} {}\n'.format(*pair)) for pair in _pair_list]               

        self.logger.info(f'Pair list saved as {self.pair_list}.')
        return 0
    
    
class ComputeMatches(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.geometric_model = 'a' if self.camera_model == 7 else prj_cfg['sfm']['feature_matching']['geometric_model']
        self.nnd_ratio = prj_cfg['sfm']['feature_matching']['nnd_ratio']
        self.match_alg = prj_cfg['sfm']['feature_matching']['match_alg']
        
        self.matches_dir = prj_dirs['matches']
        self.img_dir = prj_dirs['img']
        
        self.pair_list = join(prj_dirs['sfm'], 'pair_list.txt')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Generate pair list and then run the feature matching command. '''
        if not os.path.isfile(self.pair_list):
            self.logger.warn('No pair_list. Use default (fullpair) !')
            self.pair_list = ''

        self.logger.info('SfM feature matching started.')
        cmd = ['openMVG_main_ComputeMatches',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-o', self.matches_dir,
               '-f', str(0),
               '-r', str(self.nnd_ratio),
               '-g', self.geometric_model,
               '-l', self.pair_list,
               '-n', self.match_alg,
               '-m', str(0)]
        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM feature matching ended with return code {proc.returncode}.')
        return proc.returncode
    

class VisualizeMatches(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
                
        self.matches_dir = prj_dirs['matches']
        self.output_dir = join(prj_dirs['sfm'], 'visualize_matches')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the match visualization command. '''

        self.logger.info('SfM match visualization started.')
        cmd = ['openMVG_main_exportMatches',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-d', self.matches_dir,
               '-m', join(self.matches_dir, 'matches.f.bin'),
               '-o', self.output_dir]
        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM match visualization ended with return code {proc.returncode}.')
        return proc.returncode

    
class IncreRecon(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.calc_intrinsics = prj_cfg['sfm']['reconstruction']['calc_intrinsics']
        self.init_img = prj_cfg['sfm']['reconstruction']['init_img']
        
        self.matches_dir = prj_dirs['matches']
        self.sfm_dir = prj_dirs['sfm']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the incremental reconstruction command. '''

        self.logger.info('SfM feature computation started.')
        cmd = ['openMVG_main_IncrementalSfM',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-m', self.matches_dir,
               '-o', self.sfm_dir,
               '-c', str(self.camera_model),
               '-f', self.calc_intrinsics,
               '-t', str(3), # triangulation method (default 3: INVERSE_DEPTH_WEIGHTED_MIDPOINT)
               '-r', str(3)] # resection/pose estimation (default 3: P3P_NORDBERG_ECCV18)
        if (self.init_img[0] != '') and (self.init_img[1] != ''):
            cmd += ['-a', self.init_img[0],
                    '-b', self.init_img[1]]
        proc = run_log_cmd(self.logger, cmd)
        self.logger.info(f'SfM incremental reconstruction ended with return code {proc.returncode}.')
        if proc.returncode != 0: 
            self.logger.error('Error occured. Stop running further commands.')
            return proc.returncode
        
        # Copy the sfm data to data directory. Use copy for filtering failure recovery
        sfm_data_bin_path = shutil.copy(join(self.sfm_dir, 'sfm_data.bin'), join(self.data_dir, 'sfm_data.bin'))
        
        return proc.returncode


class IncreReconV2(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.calc_intrinsics = prj_cfg['sfm']['reconstruction']['calc_intrinsics']
        # self.init_img = prj_cfg['sfm']['reconstruction']['init_img']
        
        self.matches_dir = prj_dirs['matches']
        self.sfm_dir = prj_dirs['sfm']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the incremental reconstruction command. '''

        self.logger.info('SfM feature computation started.')
        cmd = ['openMVG_main_IncrementalSfM2',
               '-i', join(self.matches_dir, 'sfm_data.json'),
               '-m', self.matches_dir,
               '-o', self.sfm_dir,
               '-c', str(self.camera_model),
               '-S', 'STELLAR',
               '-f', self.calc_intrinsics,
               '-t', str(3), # triangulation method (default 3: INVERSE_DEPTH_WEIGHTED_MIDPOINT)
               '-r', str(3)] # resection/pose estimation (default 3: P3P_NORDBERG_ECCV18)
        # if (self.init_img[0] != '') and (self.init_img[1] != ''):
        #     cmd += ['-a', self.init_img[0],
        #             '-b', self.init_img[1]]
        proc = run_log_cmd(self.logger, cmd)
        self.logger.info(f'SfM incremental reconstruction ended with return code {proc.returncode}.')
        if proc.returncode != 0: 
            self.logger.error('Error occured. Stop running further commands.')
            return proc.returncode
        
        # Copy the sfm data to data directory. Use copy for filtering failure recovery
        sfm_data_bin_path = shutil.copy(join(self.sfm_dir, 'sfm_data.bin'), join(self.data_dir, 'sfm_data.bin'))
        
        return proc.returncode

from campose_refiner.rig_mean_campose_refiner import RigMeanCamposeRefiner
from campose_refiner.rig_center_campose_refiner import RigCenterCamposeRefiner
class RefineSfMRCampose(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])

        refiner_type = prj_cfg['sfm']['campose_refiner']['refiner_type']
        refiner_config = prj_cfg['sfm']['campose_refiner']['refiner_config']
        if refiner_type == 'rig_mean':
            self.refiner = RigMeanCamposeRefiner(refiner_config)
        elif refiner_type == 'rig_center':
            self.refiner = RigCenterCamposeRefiner(refiner_config)
        else:
            raise NotImplementedError
        
        self.sfm_dir = prj_dirs['sfm']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Refine points in sfm_data. '''
        
        self.logger.info('Refine points')

        # Covert bin to json
        self.logger.info('Covert bin to json.')
        cmd = ['openMVG_main_ConvertSfM_DataFormat',
               '-i', join(self.sfm_dir, 'sfm_data.bin'),
               '-o', join(self.sfm_dir, 'sfm_data.json')]
        proc = run_log_cmd(self.logger, cmd)
        if proc.returncode != 0: 
            self.logger.error('Stop running further commands.')
            return proc.returncode

        with open(join(self.sfm_dir, 'sfm_data.json'), 'r') as f:
            sfm_data = json.load(f)
        
        sfm_data = self.refiner.refine(sfm_data)

        with open(join(self.sfm_dir, 'sfm_data.json'), 'w') as f:
            safe_json_dump(sfm_data, f)

        # Covert json to bin
        self.logger.info('Covert json to bin.')
        cmd = ['openMVG_main_ConvertSfM_DataFormat',
               '-i', join(self.sfm_dir, 'sfm_data.json'),
               '-o', join(self.sfm_dir, 'sfm_data.bin')]
        proc = run_log_cmd(self.logger, cmd)
        if proc.returncode != 0: 
            self.logger.error('Stop running further commands.')
            return proc.returncode

        os.remove(join(self.sfm_dir, 'sfm_data.json'))
        sfm_data_bin_path = shutil.copy(join(self.sfm_dir, 'sfm_data.bin'), join(self.data_dir, 'sfm_data.bin'))

        return 0


class IncreReconRefine(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.calc_intrinsics = prj_cfg['sfm']['reconstruction']['calc_intrinsics']
        # self.init_img = prj_cfg['sfm']['reconstruction']['init_img']
        
        self.matches_dir = prj_dirs['matches']
        self.sfm_dir = prj_dirs['sfm']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Just run the incremental reconstruction command. '''

        self.logger.info('SfM feature computation started.')

        cmd = ['openMVG_main_IncrementalSfM2',
               '-i', join(self.sfm_dir, 'sfm_data.bin'),
               '-m', self.matches_dir,
               '-o', self.sfm_dir,
               '-c', str(self.camera_model),
               '-S', 'EXISTING_POSE',
               '-f', self.calc_intrinsics,
               '-t', str(3), # triangulation method (default 3: INVERSE_DEPTH_WEIGHTED_MIDPOINT)
               '-r', str(3)] # resection/pose estimation (default 3: P3P_NORDBERG_ECCV18)
        # if (self.init_img[0] != '') and (self.init_img[1] != ''):
        #     cmd += ['-a', self.init_img[0],
        #             '-b', self.init_img[1]]
        proc = run_log_cmd(self.logger, cmd)
        self.logger.info(f'SfM incremental reconstruction ended with return code {proc.returncode}.')
        if proc.returncode != 0: 
            self.logger.error('Error occured. Stop running further commands.')
            return proc.returncode
        
        # Copy the sfm data to data directory. Use copy for filtering failure recovery
        sfm_data_bin_path = shutil.copy(join(self.sfm_dir, 'sfm_data.bin'), join(self.data_dir, 'sfm_data.bin'))
        
        return proc.returncode


from point_filter.cov_point_filter import CovPointFilter

class FilterSfMResults(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])

        filter_type = prj_cfg['sfm']['filter']['filter_type']
        filter_config = prj_cfg['sfm']['filter']['filter_config']
        if filter_type == 'cov':
            self.filter = CovPointFilter(filter_config)
        else:
            raise NotImplementedError
        
        self.sfm_dir = prj_dirs['sfm']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Filter points in sfm_data. '''
        
        self.logger.info('Filter points')

        # Covert bin to json
        self.logger.info('Covert bin to json.')
        cmd = ['openMVG_main_ConvertSfM_DataFormat',
               '-i', join(self.sfm_dir, 'sfm_data.bin'),
               '-o', join(self.sfm_dir, 'sfm_data.json')]
        proc = run_log_cmd(self.logger, cmd)
        if proc.returncode != 0: 
            self.logger.error('Stop running further commands.')
            return proc.returncode

        with open(join(self.sfm_dir, 'sfm_data.json'), 'r') as f:
            sfm_data = json.load(f)
        
        structure = sfm_data['structure']
        points = np.array([s['value']['X'] for s in structure])
        valid = self.filter.filter(points)
        sfm_data['structure'] = list(itertools.compress(structure, valid))

        with open(join(self.sfm_dir, 'sfm_data_filtered.json'), 'w') as f:
            json.dump(sfm_data, f)

        # Covert json to bin
        self.logger.info('Covert json to bin.')
        cmd = ['openMVG_main_ConvertSfM_DataFormat',
               '-i', join(self.sfm_dir, 'sfm_data_filtered.json'),
               '-o', join(self.sfm_dir, 'sfm_data_filtered.bin')]
        proc = run_log_cmd(self.logger, cmd)
        if proc.returncode != 0: 
            self.logger.error('Stop running further commands.')
            return proc.returncode

        sfm_data_bin_path = shutil.copy(join(self.sfm_dir, 'sfm_data_filtered.bin'), join(self.data_dir, 'sfm_data.bin'))

        return 0


class ExportSfMResults(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.camera_model = prj_cfg['sfm']['camera_model']
        self.n_thread = prj_cfg['project']['n_thread']
        
        self.undistorted_dir = prj_dirs['undistorted']
        self.sfm_dir = prj_dirs['sfm']
        self.mvs_dir = prj_dirs['mvs']
        self.data_dir = prj_dirs['data']
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        retry = 0
        retry_max = 5
        while (retry < retry_max):
            returncode = self.run_task_content_inner()
            if returncode != 0:
                retry +=1
                time.sleep(5)
                self.logger.info(f'Retry Export_sfm_result: {retry}')
                if retry >= retry_max:
                    self.logger.error('Stop running futher commands.')
                    return returncode
            else:
                return returncode

    def run_task_content_inner(self) -> int:
        '''
        Move sfm_data.bin to data directory, then run the 3 export commands
        (all from sfm_data.bin):
            - Colorize point cloud (output: scene.ply)
            - Convert sfm data format (output: sfm_data.json)
            - Export to MVS (output: scene.mvs)
        Then create an identity transformation json.
        '''
        
        # Definition of paths
        sfm_data_bin_path = join(self.data_dir, 'sfm_data.bin')
        sfm_data_perspective_bin_path = join(self.sfm_dir, 'sfm_data_perspective.bin')  # It's only used to generate mvs related files so put it in sfm dir
        orig_sfm_data_json_path = join(self.data_dir, 'sfm_data_original.json')
        sfm_data_json_path = join(self.data_dir, 'sfm_data_transformed.json')
        transformation_json_path = join(self.data_dir, 'transformation.json')

        # ColorizePCL
        self.logger.info('SfM point cloud colorization started.')
        cmd = ['openMVG_main_ComputeSfM_DataColor',
               '-i', sfm_data_bin_path,
               '-o', join(self.sfm_dir, 'scene.ply')]
        proc = run_log_cmd(self.logger, cmd)
        self.logger.info(f'SfM point cloud colorization ended with return code {proc.returncode}.')
        if proc.returncode != 0: 
            self.logger.error('Stop running further commands.')
            return proc.returncode

        if self.camera_model == 7:
            self.logger.info('SfM conversion of result to cubemap started.')
            cmd = ['openMVG_main_openMVGSpherical2Cubic',
                   '-i', sfm_data_bin_path,
                   '-o', self.sfm_dir]
            
            
            cmd_name = 'Spherical2Cubic' 
            returncode = run_cmd_w_retry(self.logger, cmd, cmd_name, retry_max = 5)
            if returncode !=0:
                return returncode

        # ExportMVS
        self.logger.info('SfM mvs file export started.')
        cmd = ['openMVG_main_openMVG2openMVS',
               '-i', sfm_data_perspective_bin_path if (self.camera_model == 7) else sfm_data_bin_path,
               '-o', join(self.mvs_dir, 'scene.mvs'),
               '-d', self.undistorted_dir,
               '-n', str(self.n_thread)]
        
        cmd_name = 'mvg2mvs' 
        returncode = run_cmd_w_retry(self.logger, cmd, cmd_name, retry_max = 5)
        if returncode !=0:
            return returncode
        
       # ConvertSfM (Somehow has to be run after ExportMVS)
        self.logger.info('SfM data conversion started.')
        cmd = ['openMVG_main_ConvertSfM_DataFormat',
               'binary',
               '-i', sfm_data_bin_path,
               '-o', sfm_data_json_path,
               '-V',
               '-I',
               '-E']
        proc = run_log_cmd(self.logger, cmd)
        self.logger.info(f'SfM data conversion ended with return code {proc.returncode}.')
        shutil.copyfile(sfm_data_json_path, orig_sfm_data_json_path)  # Make a copy, localization project (MatchModels) will need it.
        
        # Create identity transformation json
        trans_data = TransData().set_from_input([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], note=type(self).__name__)
        trans_data.write2json(transformation_json_path)
        self.logger.info('The sfm_data.bin is moved to data directory, sfm_data_transformed.json & transformation.json are also generated there.')

        return proc.returncode


class Localize(Task):
    def run_localization(self) -> int:
        ''' Just run the localization command. '''

        self.logger.info('SfM localization started.')
        cmd = ['openMVG_main_SfM_Localization',
               '-i', self.sfm_data_path,
               '-m', self.matches_dir,
               '-o', self.loc_dir,
               '-u', self.loc_matches_dir,
               '-q', self.loc_query_dir,
               '-r', str(self.threshold),
               '-n', str(self.n_thread)]
        if self.intrinsics_sub == 'separate':
            cmd += ['-l', self.sub_sfm_data_path, '-S', 'ON']
        elif self.intrinsics_sub == 'share':
            cmd += ['-s', 'ON']
        elif self.intrinsics_sub == 'optimize':
            pass  # cmd += ['-s', 'OFF'] <- This doesn't work but not needed anyway
        else:
            self.logger.error('How to determine intrinsics is not specified. Abort.')
            raise RuntimeError

        proc = run_log_cmd(self.logger, cmd)

        self.logger.info(f'SfM localization ended with return code {proc.returncode}.')
        return proc.returncode


class MatchSub2Main(Localize):
    """
    Input: main model and sub model
    1. utilize localization feature in openMVG to localize the frames from sub model to main model
       so it takes the frames from sub model and ask main model where are those frames should be in main model
       then return sfm data model contains both original view from main model and extra views from sub model(in main model's coord. system)
       this model is exported as sfm_data_expanded.json
    
    2. the views from main model will be then removed from sfm_data_expanded.json
       and exported as sfm_data_sub2main.json

    3. reorder and filter sfm_data_sub2main.json
    

    there are two cases:
    case 1. Matching data between adjcent slices within same route
            only doing step 3.
    case 2. Matching data between two routes: follow steps 1, 2, 3
    """
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.threshold = prj_cfg['transform']['threshold']
        self.intrinsics_sub = prj_cfg['transform']['intrinsics_sub']
        
        self.loc_dir = prj_dirs['loc']
        self.loc_matches_dir = prj_dirs['matches']
        self.loc_query_dir = prj_dirs['query']
        
        self.main_data_dir = prj_dirs['main_data']
        self.matches_dir = prj_dirs['main_matches']
        self.sub_data_dir = prj_dirs['sub_data']
        self.sub_img_dir = prj_dirs['sub_img']

        self.sfm_data_path = join(prj_dirs['main_data'], 'sfm_data.bin')  # Can it be json? Or it has to be bin?
        if self.intrinsics_sub == 'separate':
            self.sub_sfm_data_path = join(prj_dirs['sub_data'], 'sfm_data.bin')
            
        
        self.prefix = 'sub_'  # Prefix for copied images from submodel
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')    

    def run_task_content(self) -> int:
        ''' Run the localization command, then postprocess the output sfm data json file. '''
        
        # Definition of file names
        #if self.opt_with_stepsize_anchor:
        #    orig_sfm_data_name = 'sfm_data_stepsize_anchor_original.json'
        #else:
        orig_sfm_data_name = 'sfm_data_original.json'
        expanded_json_name = 'sfm_data_expanded.json'
        main_scene_json_name = 'sfm_data_original_main.json'
        subscene_json_name = 'sfm_data_original_sub.json'
        sub2main_json_nofilter_name = 'sfm_data_sub2main_nofilter.json'
        sub2main_json_name = 'sfm_data_sub2main.json'
        route_name = 'routes.json'

        # Collect relevant json files from main & sub partial project
        shutil.copyfile(join(self.main_data_dir, orig_sfm_data_name), join(self.loc_dir, main_scene_json_name))
        shutil.copyfile(join(self.sub_data_dir, orig_sfm_data_name), join(self.loc_dir, subscene_json_name))
        
        # Check if localization should be run
        with open(join(self.main_data_dir, route_name), 'r') as f:
            main_route = json.load(f)
        with open(join(self.sub_data_dir, route_name), 'r') as f:
            sub_route = json.load(f)
        should_run_loc = False
        if main_route['is_scene'] == False:
            self.logger.error('Main model cannot be object. Abort.')
            raise RuntimeError
        if sub_route['is_scene'] == False:
            should_run_loc = True
            self.logger.warn('Submodel is object, localization needed.')
        else:
            if main_route['video'] != sub_route['video']:
                should_run_loc = True
                self.logger.warn('Main and subscene are from different source video, localization needed.')
            if int(np.abs(main_route['frames'][0]['slice'] - sub_route['frames'][0]['slice'])) != 1:
                should_run_loc = True
                self.logger.warn('Main and subscene are not adjacent slices, localization needed.')
        
        # If yes, don't run localization, just prepare the loc folder; if no, run localization.
        if should_run_loc == False:
            sub2main_json_path = join(self.loc_dir, sub2main_json_name) 
            shutil.copyfile(join(self.loc_dir, main_scene_json_name), sub2main_json_path)  # TODO (hsien): Only copy the frames in window
            shutil.rmtree(self.loc_matches_dir)
            shutil.rmtree(self.loc_query_dir)
            self.logger.warn('Since main and sub are adjacent slices, no localization is needed. Just make sfm_data_sub2main.json.')
            #reorder sub2main_json
            reorder_sfm_data(join(self.loc_dir, sub2main_json_name), \
                         join(self.loc_dir, sub2main_json_name), self.logger)
            
        else:
            self.logger.info('Use localization command to match two models.')

            # Delete old query if it exists
            if os.path.isdir(self.loc_query_dir):
                self.logger.warn(f'The localization query folder {self.loc_query_dir} already exists and will be deleted.')
                shutil.rmtree(self.loc_query_dir)
            # Copy query images of submodel and rename to avoid conflict with images of main model
            shutil.copytree(self.sub_img_dir, self.loc_query_dir, symlinks=False)
            for name in os.listdir(self.loc_query_dir):
                if name.split('.')[-1] == 'jpg':
                    new_name = self.prefix + name
                    os.rename(join(self.loc_query_dir, name), join(self.loc_query_dir, new_name))

            # Run localization
            returncode = self.run_localization()
            if returncode != 0:
                self.logger.error('Localization ended but failed.')
                return returncode
            self.logger.info('Finished sub-to-main localization, start postprocessing of the sfm data file...')
            
            # Substitute query images with symlinks to save space
            shutil.rmtree(self.loc_query_dir)
            shutil.copytree(self.sub_img_dir, self.loc_query_dir, symlinks=True)

            # Remove entries in expanded json that are already in main json
            sfm_data_main_lengths = {}
            # Get the number of views & extrinsics in main sfm data
            with open(join(self.loc_dir, main_scene_json_name), 'r') as f:
                main_dict = json.load(f)
                n_view_main = len(main_dict['views'])
                n_extr_main = len(main_dict['extrinsics'])
            # Load sub2main sfm data and delete the first n elements in views & extrinsics
            # (sub2main things are all appended in back of main things)
            with open(join(self.loc_dir, expanded_json_name), 'r') as f:
                sub2main_dict = json.load(f)
                sub2main_dict['views'] = sub2main_dict['views'][n_view_main:]
                sub2main_dict['extrinsics'] = sub2main_dict['extrinsics'][n_extr_main:]
                # Shift key number correspondingly and remove prefix
                for i, _ in enumerate(sub2main_dict['views']):
                    self.logger.info(sub2main_dict['views'][i]['value']['ptr_wrapper']['data']['filename'])
                    sub2main_dict['views'][i]['value']['ptr_wrapper']['data']['filename'] = \
                        sub2main_dict['views'][i]['value']['ptr_wrapper']['data']['filename'].replace(self.prefix, '')
                    self.logger.info(sub2main_dict['views'][i]['value']['ptr_wrapper']['data']['filename'])
                    sub2main_dict['views'][i]['key'] -= n_view_main
                for i, _ in enumerate(sub2main_dict['extrinsics']):
                    sub2main_dict['extrinsics'][i]['key'] -= n_view_main
                sub2main_dict.pop('root_path', None)
                sub2main_dict['main_data_dir'] = self.main_data_dir
                sub2main_dict['sub_sfm_dir'] = self.sub_data_dir
            length_extrinsics = len(sub2main_dict['extrinsics'])
            self.logger.info(f'length of extrinsics: {length_extrinsics}')
            if length_extrinsics == 0:
                self.logger.info(f'lenghth of extrinsics: {length_extrinsics}')
                return -1
            
            with open(join(self.loc_dir, sub2main_json_nofilter_name), 'w') as f:
                json.dump(sub2main_dict, f)
            
            shutil.copyfile(join(self.loc_dir, sub2main_json_nofilter_name), \
                            join(self.loc_dir, sub2main_json_name))

            export_sfm_data_visualization(join(self.loc_dir, sub2main_json_nofilter_name), self.logger)

            self.logger.info(f'Finished postprocessing of the sfm data file and saved as {sub2main_json_name}.')
            #reorder sub2main_json
            reorder_sfm_data(join(self.loc_dir, sub2main_json_name), \
                         join(self.loc_dir, sub2main_json_name), self.logger)
            #fileter sub2main_json(only filter data if runs localization)
            filter_sfm_data(join(self.loc_dir, sub2main_json_name), \
                            join(self.loc_dir, sub2main_json_name), self.logger)

        return 0

        
class ExpandFPS(Localize):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        self.n_thread = prj_cfg['project']['n_thread']
        self.threshold = prj_cfg['sfm']['expand_fps']['threshold']
        self.intrinsics_sub = prj_cfg['sfm']['expand_fps']['intrinsics_sub'] # 'separate' 
        if self.intrinsics_sub == 'separate':
            self.loc_camera_model = prj_cfg['sfm']['loc_camera_model']
            self.loc_focal_length = prj_cfg['sfm']['loc_focal_length']
        
        self.matches_dir = prj_dirs['matches']
        self.loc_dir = prj_dirs['loc']
        self.loc_matches_dir = prj_dirs['loc_matches']
        self.loc_query_dir = prj_dirs['img_upsmpl']
        self.data_dir = prj_dirs['data']

        self.sfm_data_path = join(self.data_dir, 'sfm_data.bin')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
        
    def run_task_content(self) -> int:
        ''' Run the localization command, then reorder the sequence of the output sfm data json file. '''

        if self.intrinsics_sub == 'separate':
            cmd = ['openMVG_main_SfMInit_ImageListing',
               '-i', self.loc_query_dir,
               '-o', self.loc_matches_dir,
               '-c', str(self.loc_camera_model),
               '-f', str(self.loc_focal_length)]        
            proc = run_log_cmd(self.logger, cmd)
            self.sub_sfm_data_path = join(self.loc_matches_dir, 'sfm_data.json')
        
        input_json_path = join(self.loc_dir, 'sfm_data_expanded.json')
        output_json_path = join(self.loc_dir, 'sfm_data_expanded_reordered.json')
        final_json_path = join(self.data_dir, 'sfm_data_transformed.json')
        backup_json_path = join(self.data_dir, 'sfm_data_expandFPS.json')
        
        self.logger.info('Use localization command to expand FPS.')
        returncode = self.run_localization()
        if returncode != 0:
            self.logger.error('Localization ended but failed, so stop resorting the expanded sfm_data json file.')
            return returncode
        
        # Reorder the sequence of views and extrinsics according to image file names
        self.logger.info(f'Start reordering {input_json_path}...')
        with open(input_json_path, 'r') as f:
            data = json.load(f)

            # === Make a image file numbering list with original order ===
            # Image name: imgNr_lenseNr_hash.xxx
            img_name_str = [
                re.split(r'[.]', view['value']['ptr_wrapper']['data']['filename'])[0]
                for view in data['views']]
            # Sort according to the above list and extract the new order
            seq = [element[0] for element in sorted(enumerate(img_name_str), key=lambda x:x[1])]

            # === Rearrange the views and extrincs according to the new order ===
            # Make a extrinsics dict with ['key'] as key
            extrinsics_dict = {}
            for extr in data['extrinsics']:
                extrinsics_dict[extr['key']] = extr['value']
            # Place the view / extr pair into a new dict w/ the new order, if no pair then skip the view
            reordered_data = {'views':[], 'extrinsics':[]}
            for i in seq:
                view = data['views'][i]
                try:
                    reordered_data['extrinsics'].append(
                        {'key': view['key'], 'value': extrinsics_dict[view['key']]})
                    reordered_data['views'].append(view)
                except Exception as e:
                    self.logger.warning(f'Failed to find view / extr pair using key nr.{view["key"]}. Skip it. Original error message: {e}')
            
            # Put reordered things back and dump 
            data['views'] = reordered_data['views']
            data['extrinsics'] = reordered_data['extrinsics']

            json.dump(data, open(output_json_path, 'w'))

        self.logger.info(f'Finish reordering {input_json_path}...')

        # Copy json to data dir
        shutil.copyfile(output_json_path, final_json_path)
        shutil.copyfile(output_json_path, backup_json_path)
        
        return returncode
