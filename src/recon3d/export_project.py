#!/opt/conda/bin/python

import os, sys
import pickle 
import shutil
from project import PreprocProj, ReconProj, LocProj, ExportProj
import json, yaml
from utils.safe_yaml import safe_yaml_load, safe_yaml_dump
from utils.exportutil import copy_2_tower, copy_from_tower
from utils.logger import get_logger, close_all_handlers
from utils.src_marker import read_src_marker, get_src_marker_info
from super_project import find_filename_w_label


class ExportProject:
    '''
    ExperProject is to lauch a export project for single tower repository.
    Export project use a GPS geojson as anchor, find the coordinate transformation 
    from SfM to GPS, and export result from SfM to geo_data_sfm.geojson
    Form a tower repo that has been through:
    1. Preprocess
    2. Reconstruction
    3. Localization
    It will continue to run
    4. Export
    Input configs:
    project:
        type: 'ExportProj'
        name: str, project name
        project_dir: str, project root path
        force_new_project_dir: bool, if old folder should be deltedted and create a new one
        n_thread: int, number of threads
        process_prio: 2, process_priority in openMVS
        n_lense_for_rot: lense ID for merged rotation
        pipeline:
            - 'CONCAT_CONVERT_TO_GEOJSON_GPS_ANCHORED'
    geojson:
        remake_routes: bool, if routes.json should be rewritten
        anchor_pts_num: int, number of points used in GPS geojson as anchor points
        anchor_gps: str, file path of anchor GPS geosjon
        gps_combined: bool, if SfM routes should combined with GPS anchor geojson
        smoothen_match: bool, if matched anchored area should be horizontally smoothened 
        smooth_type: str, 'anchor' or 'all'
        match_footprint: bool, if footprint should join find transformation process
    The input config was obtained via 
    exp_cfg_src, exp_cfg_template.
    exp_cfg_src will be changed for each Export job of different tower repo,
    exp_cfg_template is imported from /sfm-lab/src/config/scene_v0_hard_gopro_template_export.yaml
    
    Args: 
    exp_cfg_src: ExportProjectSrc
    exp_cfg_template: dict read from template file
    '''
    def __init__(self, exp_cfg_src, exp_cfg_template, logger_dir='local', logger_name = 'ExportProject.log'):
        self.export_cfg = self.get_config(exp_cfg_src, exp_cfg_template)
        
       
        if not os.path.exists(self.export_cfg['export_config_template']['project']['project_dir']):
            os.makedirs(self.export_cfg['export_config_template']['project']['project_dir'])
        
        if logger_dir == 'local':
            self.logger = get_logger(logger_name, \
                    self.export_cfg['export_config_template']['project']['project_dir'])        
        else:
            self.logger = get_logger(logger_name, logger_dir)        
        
    @staticmethod
    def get_config(export_cfg_src, export_cfg):
        '''
        assign config info from export_cfg_src to export_cfg
        Args:
        export_cfg_src: dict of src config
        export_cfg: dict of template config 
        Return:
        export_cfg: dict of full config of ExportProject
        '''
        export_cfg['parent_dir'] = export_cfg_src['parent_dir']
        export_cfg['name'] = export_cfg_src['name']
        export_cfg['tower_src_dir'] = export_cfg_src['tower_src_dir']
        export_cfg['src_mrk_start_sec'] = export_cfg_src['src_mrk_start_sec']
        export_cfg['use_cutted_video'] = export_cfg_src['use_cutted_video']
        
        export_cfg['export_config_template']['project']['tower_src_dir'] =\
                export_cfg_src['tower_src_dir']
        
        export_cfg['export_config_template']['geojson']['anchor_gps'] = export_cfg_src['anchor_gps']
        export_cfg['export_config_template']['geojson']['src_mrk_start_sec'] = export_cfg_src['src_mrk_start_sec']
        export_cfg['export_config_template']['geojson']['use_cutted_video'] = export_cfg_src['use_cutted_video']
        
        proj_path = os.path.join(export_cfg_src['parent_dir'], export_cfg_src['name'])
        export_cfg['export_config_template']['project']['project_dir'] = os.path.join(proj_path, 'export')
        export_cfg['export_config_template']['project']['name'] = export_cfg['name']
 
        return export_cfg 
    
    def run(self):
        '''
        prepare data folder and launch ExportProject
        1. copy data folder that contains SfM results and gps anchor file from tower repo
        2. launch ExportProj in local workspace
        '''
        # Copy data from Tower
        if copy_from_tower(self.export_cfg['export_config_template']['project']['project_dir'], \
                self.export_cfg['tower_src_dir']) !=0:
            self.logger.error('Copy from Tower failed')
            raise RuntimeError('Copy from Tower failed')

        # Run export proj
        if ExportProj(self.export_cfg['export_config_template']).run_pipeline() != 0:
            self.logger.error('ExportProj pipeline failed.')
            raise RuntimeError('ExportProj pipeline failed.')
        else:
            self.logger.info('ExportProj pipeline passed.')
            
        self.logger.info('ExportProject ends successfully')

class ExportProjectSrc():
    '''
    create input dict for export project 
    it will parse necessary information from tower repository 
    and feed into ExportProject
    Args: 
    tower_dir: tower repo directory
    src_marker: filename source marker file
    gps_track: filename of gps.geojson file
    proj_dir_root: directory that a SfM project will be created
    '''
    def __init__(self, tower_dir, tower_dir_anchor, src_marker, gps_track, proj_dir_root, use_cutted_video):
        self.tower_dir = tower_dir
        self.src_marker_file = os.path.join(self.tower_dir, src_marker)
        self.proj_dir = proj_dir_root
        self.proj_name = self.tower_dir.split('/')[-1]
        self.src_marker_dict = \
                get_src_marker_info(read_src_marker(self.src_marker_file))
        self.gps_track = os.path.join(tower_dir_anchor, gps_track)
        self.export_src_dict = self.gen_export_src_dict(use_cutted_video) 
   
    def get_export_src_dict(self):
        return self.export_src_dict
    
    def gen_export_src_dict(self, use_cutted_video):
        '''
        create necessary entries for input of export project
        name: the name of folder that holds SfM project
        parent_dir: the location where SfM will be created. 
                    ex: /aaa/bbb/ccc
                    the full project path will be /aaa/bbb/ccc/name
        tower_src_dir: tower repository
        anchor_gps: the file that will be used as GPS anchor
        '''
        export_src_dict = {}
        export_src_dict['name'] = self.proj_name
        export_src_dict['parent_dir'] = self.proj_dir
        export_src_dict['tower_src_dir'] = self.tower_dir
        export_src_dict['anchor_gps'] = self.gps_track
        export_src_dict['src_mrk_start_sec'] = self.src_marker_dict['ss']
        export_src_dict['use_cutted_video'] = use_cutted_video
        return export_src_dict


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--tower_dir", required=True, help='tower repo directory')
    parser.add_argument("-g", "--gps_tower_dir", required=True, help='tower repo directory of gps anchor')
    parser.add_argument("-f", "--template_file", required=True, help='template file')
    parser.add_argument("-r", "--proj_dir_root", required=True, help='project root dir')
    parser.add_argument("-u", "--use_cutted_video", action="store_true")
    
    input_vals = parser.parse_args()
    tower_dir = input_vals.tower_dir
    gps_tower_dir = input_vals.gps_tower_dir
    template_file = input_vals.template_file
    proj_dir_root = input_vals.proj_dir_root
    video_path = find_filename_w_label(tower_dir, '@stitched-video')
    src_marker = find_filename_w_label(tower_dir, '@marker')
    gps_track  = find_filename_w_label(gps_tower_dir, '@gpstrack')
    
    use_cutted_video = input_vals.use_cutted_video
    export_cfg_src = ExportProjectSrc(tower_dir, gps_tower_dir, \
                                      src_marker, gps_track, \
                                      proj_dir_root, use_cutted_video).get_export_src_dict() 
    export_cfg_template = safe_yaml_load(template_file)

    ExportProject(export_cfg_src, export_cfg_template).run()
