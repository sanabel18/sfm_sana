#!/opt/conda/bin/python
import os, sys
import pickle 
import shutil
from project import PreprocProj, ReconProj, LocProj
import json, yaml
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump
from utils.exportutil import copy_2_tower
from utils.logger import get_logger, close_all_handlers
from utils.src_marker import read_src_marker, get_src_marker_info 
from super_project_src_tower import SuperProjectSrcTower

class SuperProject:
    '''
    SuperProject is to lauch a SfM pipeline for single tower repository.
    It will run 3 kinds of Projects:
    1. Preprocess
    2. Reconstruction
    3. Localization
    The input of super project has two main parts:
    super_cfg_src:
    it is the part that will change for every tower repo, therefore
    needs to be specified each time super project was launched.
    it should contain super_src_dict that has keys:
        super_src_dict['meta']: 'SCENE_V0' # specifying what kind of super project
        super_src_dict['name']: project name # name of SfM project
        super_src_dict['parent_dir']: # path of SfM prject root
        super_src_dict['ss']: start time of video
        super_src_dict['to']: end time of video
        super_src_dict['tower_src_dir']: path of tower repository 
        super_src_dict['anchor_pts_num']: 1000 # number of pts in GPS anchor
        super_src_dict['anchor_gps']: filename of GPS anchor
        super_src_dict['video_path']: filename of video usded for SfM
        super_src_dict['key_frame_idx_file']: key frame index npy file path
    super_cfg_template:
        the parameter to be used within each tasks in projects 1~4. 
        the example can be found at 
        /sfm-lab/src/config/scene_v0_hard_gopro_template_export_refine.yaml
    
    After 4 projects are all finished successfully, the reconstructed_GPS.geojson 
    and data_* files will be copied to tower repository
    '''
    def __init__(self, super_cfg_src_file, super_cfg_template_file, logger_dir='local', logger_name = 'SuperProject.log'):
       
        

        self.super_cfg = SuperProject.get_config(super_cfg_src_file, super_cfg_template_file)

        self.dirs = dict()
        self.dirs['prj'] = os.path.join(self.super_cfg['parent_dir'], self.super_cfg['name'])
        os.makedirs(self.dirs['prj'], exist_ok=True)
        
        if logger_dir == 'local':
            self.logger = get_logger(logger_name, self.dirs['prj'])        
        else:
            self.logger = get_logger(logger_name, logger_dir)        
        
        self.dirs['preproc'] = os.path.join(self.dirs['prj'], 'preproc')
        os.makedirs(self.dirs['preproc'], exist_ok=True)
        
        self.dirs['recon'] = os.path.join(self.dirs['prj'], 'recon')
        os.makedirs(self.dirs['recon'], exist_ok=True )
        
        self.dirs['loc'] = os.path.join(self.dirs['prj'], 'loc')
        os.makedirs(self.dirs['loc'], exist_ok=True)

        # === Fill in everything that should be done by super project ===
        # Preproc
        self.super_cfg['preproc_config_template']['project']['project_dir'] = self.dirs['preproc']
        # Recon
        self.super_cfg['recon_config_template']['project']['project_dir'] = self.dirs['recon']
        self.super_cfg['recon_config_template']['project']['name'] = self.super_cfg['name']
        # Loc
        if 'loc_config_template' in self.super_cfg:
            self.super_cfg['loc_config_template']['project']['project_dir'] = self.dirs['loc']
            self.super_cfg['loc_config_template']['project']['name'] = self.super_cfg['name']
            if 'tower_src_dir' in self.super_cfg:
                self.super_cfg['loc_config_template']['project']['tower_src_dir'] = self.super_cfg['tower_src_dir']
  
        # === Arrange preproc config ===
        self.preproc_cfg = self.super_cfg['preproc_config_template']
        # PreprocPrj will use recon & loc config template to generate all "real" recon & loc config files
        self.preproc_cfg['recon_config_template'] = self.super_cfg['recon_config_template']
        if 'loc_config_template' in self.super_cfg:
            self.preproc_cfg['loc_config_template'] = self.super_cfg['loc_config_template']
        if 'export_config_template' in self.super_cfg:
            self.preproc_cfg['export_config_template'] = self.super_cfg['export_config_template'] 

        # === Make a copy of super config
        #shutil.copy(super_cfg_src_file, self.super_cfg['tower_src_dir'])
        if 'tower_src_dir' in self.super_cfg:
            shutil.copy(super_cfg_template_file, self.super_cfg['tower_src_dir'])


    def safe_copy_data_dir(self, src, dst):
        os.makedirs(dst, exist_ok=True)
        
        for item in os.listdir(src):
            item_is_folder = os.path.isdir(item)
            if item_is_folder:
                if (item == 'img_prj') or (item == 'matches'):
                    print(f'Making symbolic link for {item} from {src} to {dst}...')
                    dst_item = os.path.join(dst, item)
                    src_item = os.path.join(src, item)
                    if (os.path.exists(dst_item) and os.path.islink(dst_item)):
                        os.remove(dst_item)
                    os.symlink(src_item, dst_item)  # To save space
            elif item == '.ipynb_checkpoints':
                print(f'Ignored {item}.')
            else:
                try:
                    print(f'Copying {item} from {src} to {dst}...')
                    shutil.copy(os.path.join(src, item), os.path.join(dst, item))  # Except for img and matches, everything is file not folder
                except:
                    print(f'Unexpected folder {item} is in sub data directory. Ignored.')
        return
 
    @staticmethod
    def get_config(super_cfg_src_file, super_cfg_template_file):
        super_cfg_src = safe_yaml_load(super_cfg_src_file)
        super_cfg = safe_yaml_load(super_cfg_template_file)

        assert super_cfg_src['meta'] == super_cfg['meta']

        if 'tower_src_dir' in super_cfg_src:
            super_cfg['tower_src_dir'] = super_cfg_src['tower_src_dir']
            src_path = super_cfg_src['tower_src_dir']
        else:
            src_path = super_cfg_src['src_dir']
 

        if super_cfg_src['meta'] == 'SCENE_V0':
            super_cfg['parent_dir'] = super_cfg_src['parent_dir']
            super_cfg['name'] = super_cfg_src['name']
           
            super_cfg['preproc_config_template']['preprocess_loc_config']['src_config']['src_dir'] = src_path
            super_cfg['preproc_config_template']['preprocess_loc_config']['src_config']['ss'] = super_cfg_src['ss']
            super_cfg['preproc_config_template']['preprocess_loc_config']['src_config']['to'] = super_cfg_src['to']
            if 'loc_fps' in super_cfg_src:
                super_cfg['preproc_config_template']['preprocess_loc_config']['src_config']['fps'] = super_cfg_src['loc_fps']
            super_cfg['preproc_config_template']['preprocess_loc_config']['src_config']['video_path'] = super_cfg_src['video_path']

            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['src_dir'] = src_path
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['ss'] = super_cfg_src['ss']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['to'] = super_cfg_src['to']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['fps'] = super_cfg_src['recon_fps']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['video_path'] = super_cfg_src['video_path']
            
            if 'key_frame_idx_file' in super_cfg_src:
                if super_cfg_src['key_frame_idx_file']:
                    super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['key_frame_idx_file'] = super_cfg_src['key_frame_idx_file']

        elif super_cfg_src['meta'] == 'OBJECT_IMAGE_V0':
            super_cfg['parent_dir'] = super_cfg_src['parent_dir']
            super_cfg['name'] = super_cfg_src['name']
            
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['src_dir'] = os.path.join(src_path, super_cfg_src['img_dir'])
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['focal_length'] = super_cfg_src['focal_length']

        elif super_cfg_src['meta'] == 'OBJECT_VIDEO_V0':
            super_cfg['parent_dir'] = super_cfg_src['parent_dir']
            super_cfg['name'] = super_cfg_src['name']

            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['focal_length'] = super_cfg_src['focal_length']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['src_video_list'] = super_cfg_src['src_video_list']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['ss_list'] = super_cfg_src['ss_list']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['to_list'] = super_cfg_src['to_list']
            super_cfg['preproc_config_template']['preprocess_recon_config']['src_config']['fps_list'] = super_cfg_src['fps_list']

        else:
            raise NotImplementedError

        return super_cfg


    def run(self):
        if PreprocProj(self.preproc_cfg).run_pipeline() != 0:
            self.logger.error('PreprocProj pipeline failed.')
            raise RuntimeError('PreprocProj pipeline failed.')
        else:
            self.logger.info('PreprocProj pipeline passed.')
        with open(os.path.join(self.dirs['preproc'], 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        is_slicing = False
        if 'is_slicing' in metadata:
            is_slicing = metadata['is_slicing']

        if is_slicing:
            recon_config_list = []
            for slice_dir_idx in sorted(metadata['slice_dir_dict']):
                slice_dir = metadata['slice_dir_dict'][slice_dir_idx]
                recon_config_file = os.path.join(slice_dir, 'recon_config.yaml')
                recon_config = safe_yaml_load(recon_config_file)
                recon_config_list.append(recon_config)
                if ReconProj(recon_config).run_pipeline() != 0:
                    self.logger.error('ReconProj pipeline failed.')
                    raise RuntimeError('ReconProj pipeline failed.')
                else:
                    self.logger.info('ReconProj pipeline passed.')
            for slice_dir_idx in sorted(metadata['slice_dir_dict']):
                recon_config = recon_config_list[slice_dir_idx]
                proj_dir = recon_config['project']['project_dir']
                data_folder = 'data_'+recon_config['project']['name']
                loc_dir = os.path.join(self.dirs['loc'], data_folder)
                src_dir = os.path.join(proj_dir, data_folder)
                if slice_dir_idx == 0:
                    if len(sorted(metadata['slice_dir_dict'])) == 1: 
                        self.logger.info('Only one slice, no LocProj needed.')
                        if 'tower_src_dir' in self.super_cfg:
                            if copy_2_tower(proj_dir,
                                self.super_cfg['tower_src_dir'],
                                recon_config['project']['name']) != 0:
                                self.logger.error('Copy to Tower failed.')
                                raise RuntimeError('Copy to Tower failed.')
                    continue # No localization for "head"
                
                slice_dir = metadata['slice_dir_dict'][slice_dir_idx]
                loc_config_file = os.path.join(slice_dir, 'loc_config.yaml')
                loc_config = safe_yaml_load(loc_config_file)
                if LocProj(loc_config).run_pipeline() != 0:
                    self.logger.error('LocProj pipeline failed.')
                    raise RuntimeError('LocProj pipeline failed.')
                else:
                    # copy slice_000 to loc
                    # this step ought to be done after loc_001_000 is done
                    if slice_dir_idx == 1:
                        recon_config = recon_config_list[slice_dir_idx-1]
                        proj_dir = recon_config['project']['project_dir']
                        data_folder = 'data_'+recon_config['project']['name']
                        loc_dir_000 = os.path.join(self.dirs['loc'], data_folder)
                        src_dir_000 = os.path.join(proj_dir, data_folder)
                        self.safe_copy_data_dir(src_dir_000, loc_dir_000)
                    self.logger.info('LocProj pipeline passed.')
            
            if len(sorted(metadata['slice_dir_dict'])) > 1:
                if 'tower_src_dir' in self.super_cfg:
                    if copy_2_tower(loc_config['project']['project_dir'],
                                    loc_config['project']['tower_src_dir'],
                                    loc_config['project']['name']) != 0:
                        self.logger.error('Copy to Tower failed.')
                        raise RuntimeError('Copy to Tower failed.')
                
            self.logger.info('SuperProj ends successfully')
            close_all_handlers(self.logger)
        else:
            recon_config_file = os.path.join(self.dirs['preproc'], 'recon_config.yaml')
            recon_config = safe_yaml_load(recon_config_file)
            returncode = ReconProj(recon_config).run_pipeline()
            if returncode != 0:
                self.logger.error('ReconProj failed.')
                raise RuntimeError
            self.logger.info('SuperProj ends successfully')
            close_all_handlers(self.logger)

def find_filenames_w_label(tower_dir, file_label):
    '''
    finding the filename with label file_label
    within a tower repo
    Args: 
    str: path of tower repository
    file_label: tower file label 
    Return: 
    str: file name with tower label file_label
    '''
    from tower import Tower
    tower = Tower(tower_dir)
    labels = tower.list_file_labels("")
    filenames = []
    for label in labels:
        if file_label in label['labels']:
            filenames.append(label['path'])
    return filenames

def find_filename_w_label(tower_dir, file_label):
    filenames = find_filenames_w_label(tower_dir, file_label)
    if len(filenames) > 1:
         raise ValueError(f'label {file_label} has multiple files. Should only be one.')
    else:
         return filenames[0]
