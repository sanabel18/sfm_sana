from abc import ABC, abstractmethod
import math
import os
import shutil
import json
from utils.safe_json import safe_json_dump
import toml
import yaml
from utils.safe_yaml import safe_yaml_dump
import pickle
import copy

import cv2
import numpy as np

from os.path import join
from task import Task
from utils.genutil import run_log_cmd, safely_create_dir, get_current_time_str
from processor.pipeline import *
from preprocessing.stage import *
from preprocessing.utils import get_image_file_dict
from person_mask.person_mask_model import PersonMaskModel
from person_mask.person_mask_stage import *


class PreprocessTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
        
        if self.mode == 'loc':
            self.preprocess_config = prj_cfg['preprocess_loc_config']
            self.img_dir = prj_dirs['img_upsmpl']
        elif self.mode == 'recon':
            self.preprocess_config = prj_cfg['preprocess_recon_config']
            self.img_dir = prj_dirs['img']
        elif self.mode == 'export':
            self.preprocess_config = prj_cfg['preprocess_export_config']
        else:
            error_msg = 'mode not valid'
            raise ValueError(error_msg)
        self.metadata_src_mode = self.preprocess_config['src_mode']
        # src_mode:
        # 1. 'no_order': all images are of no order
        # 2. 'videoid_frameid': all images are named by [video id]_[frame id].jpg
        # 3. 'frameid_camid': all images are named by [frame id]_[camera id].jpg
        self.prj_dir = prj_dirs['prj']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')

        src_type = self.preprocess_config['src_type']
        self.src_config = self.preprocess_config['src_config']
        if src_type == 'image':
            self.src_gen = ImageDirSourceGenerator()
        elif src_type == 'obj_video_list':
            self.src_gen = ObjVideoSourceGenerator()
        elif src_type == 'video':
            self.src_gen = SingleVideoSourceGenerator()
        elif src_type == 'insta360':
            self.src_gen = Insta360SourceGenerator()
        elif src_type == 'insta360equirect':
            self.src_gen = Insta360EquirectSourceGenerator()
        elif src_type == 'gopro_equirect':
            if 'key_frame_idx_file' in self.preprocess_config['src_config']:
                self.src_gen = GoProEquirectSourceGenerator_w_keyIdx()
            else:
                self.src_gen = GoProEquirectSourceGenerator()
        else:
            raise NotImplementedError

        self.stage_list = []
        self.stage_config_list = []
        for stage in self.preprocess_config['stage_list']:
            stage_type = stage['stage_type']
            stage_config = stage['stage_config']
            if stage_type == 'RotateStage':
                stage_processing_handler = RotateStage()
            elif stage_type == 'DefisheyeStage':
                stage_processing_handler = DefisheyeStage()
            elif stage_type == 'DownsizeStage':
                stage_processing_handler = DownsizeStage()
            elif stage_type == 'CubemapStage':
                stage_processing_handler = CubemapStage()
            elif stage_type == 'DenseCubemapStage':
                stage_processing_handler = DenseCubemapStage()
            else:
                raise NotImplementedError
            self.stage_list.append(
                {
                    'name': stage_type,
                    'stage_processing_handler': stage_processing_handler,
                    'BATCH_SIZE': 16, 
                    'FEED_SIZE_LIMIT': 32
                }
            )
            self.stage_config_list.append(stage_config)
        stage_processing_handler = ImageHashFilenameStage()
        stage_config = {}
        self.stage_list.append(
            {
                'name': 'hash',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)
        stage_processing_handler = SaveImageStage()
        stage_config = {'base_path': self.img_dir}
        self.stage_list.append(
            {
                'name': 'save',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)

        assert len(self.stage_list) == len(self.stage_config_list)
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
    
    
    def run_task_content(self):
        self.logger.info('Start running task content...')

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            self.metadata = dict()
        self._metadata = dict()
        self.src_gen.set(self.src_config, self.logger, self._metadata)

        pipeline_config = {
            'FEED_SIZE_LIMIT': 32,
            'stage_config': self.stage_list
        }
        self.pipeline = Pipeline()
        self.pipeline.build_pipeline(pipeline_config)
        input_handler = NoOpInputHandler()
        self.pipeline.set_input_handler(input_handler)
        output_handler = NoOpOutputHandler()
        self.pipeline.set_output_handler(output_handler)
        input_handler.set({})
        for stage, stage_config in zip(self.stage_list, self.stage_config_list):
            stage['stage_processing_handler'].set(stage_config, self.logger, self._metadata)
        output_handler.set({})

        self.pipeline.feed_sync(self.src_gen.gen())

        if self.mode == 'loc':
            self.metadata['loc_metadata'] = self._metadata
        elif self.mode == 'recon':
            self.metadata['recon_metadata'] = self._metadata

        if self.metadata_src_mode == 'no_order':
            # currently don't care mask
            if self.mode == 'loc':
                self.metadata['image_upsmpl_file_list'] = os.listdir(self.img_dir)
            elif self.mode == 'recon':
                self.metadata['image_file_list'] = os.listdir(self.img_dir)
        elif self.metadata_src_mode == 'videoid_frameid':
            if self.mode == 'loc':
                image_upsmpl_file_dict, image_upsmpl_file_mask_dict, video_upsmpl_idx_list = get_image_file_dict(self.img_dir)
                self.metadata['image_upsmpl_file_dict'] = image_upsmpl_file_dict
                self.metadata['image_upsmpl_file_mask_dict'] = image_upsmpl_file_mask_dict
                self.metadata['video_upsmpl_idx_list'] = video_upsmpl_idx_list
            elif self.mode == 'recon':
                image_file_dict, image_file_mask_dict, video_idx_list = get_image_file_dict(self.img_dir)
                self.metadata['image_file_dict'] = image_file_dict
                self.metadata['image_file_mask_dict'] = image_file_mask_dict
                self.metadata['video_idx_list'] = video_idx_list
        elif self.metadata_src_mode == 'frameid_camid':
            if self.mode == 'loc':
                image_upsmpl_file_dict, image_upsmpl_file_mask_dict, frame_upsmpl_idx_list = get_image_file_dict(self.img_dir)
                self.metadata['image_upsmpl_file_dict'] = image_upsmpl_file_dict
                self.metadata['image_upsmpl_file_mask_dict'] = image_upsmpl_file_mask_dict
                self.metadata['frame_upsmpl_idx_list'] = frame_upsmpl_idx_list
            elif self.mode == 'recon':
                image_file_dict, image_file_mask_dict, frame_idx_list = get_image_file_dict(self.img_dir)
                self.metadata['image_file_dict'] = image_file_dict
                self.metadata['image_file_mask_dict'] = image_file_mask_dict
                self.metadata['frame_idx_list'] = frame_idx_list

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Image preprocessing done.')

        return 0


class PreprocessLocTask(PreprocessTask):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        self.mode = 'loc'
        super().__init__(step, prj_dirs, prj_cfg)


class PreprocessReconTask(PreprocessTask):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        self.mode = 'recon'
        super().__init__(step, prj_dirs, prj_cfg)


class AllSamplingTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])

        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        # clean img_dir, workaround
        shutil.rmtree(self.img_dir) 

        os.symlink(self.img_upsmpl_dir, self.img_dir)

        if 'image_upsmpl_file_list' in self.metadata:
            self.metadata['image_file_list'] = self.metadata['image_upsmpl_file_list']
        if 'image_upsmpl_file_dict' in self.metadata:
            self.metadata['image_file_dict'] = self.metadata['image_upsmpl_file_dict']
        if 'image_upsmpl_file_mask_dict' in self.metadata:
            self.metadata['image_file_mask_dict'] = self.metadata['image_upsmpl_file_mask_dict']
        if 'video_upsmpl_idx_list' in self.metadata:
            self.metadata['video_idx_list'] = self.metadata['video_upsmpl_idx_list']
        if 'image_upsmpl_file_dict' in self.metadata:
            self.metadata['frame_idx_list'] = self.metadata['frame_upsmpl_idx_list']

        self.metadata['recon_metadata'] = self.metadata['loc_metadata']

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Sampling done.')
        return 0


class VideoSamplingTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
    
        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
    
    @abstractmethod
    def sample_from_video(self, video_file_list) -> list:
        pass
        # return list of filename to ln from img_upsmpl_dir to img_dir

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        image_upsmpl_file_dict = self.metadata['image_upsmpl_file_dict']
        image_file_dict = dict()
        # image_upsmpl_file_mask_dict = None
        # image_file_mask_dict = None
        # if 'image_upsmpl_file_mask_dict' in self.metadata:
        #     image_upsmpl_file_mask_dict = self.metadata['image_upsmpl_file_mask_dict']
        #     image_file_mask_dict = dict()


        # clean img_dir, workaround
        shutil.rmtree(self.img_dir) 
        os.mkdir(self.img_dir)

        for video_idx in self.metadata['video_upsmpl_idx_list']:
            sampled_video_file_list = self.sample_from_video(image_upsmpl_file_dict[video_idx])
            image_file_dict[video_idx] = sampled_video_file_list
            [os.symlink(join(self.img_upsmpl_dir, filename), join(self.img_dir, filename)) for filename in sampled_video_file_list]

        self.metadata['video_idx_list'] = self.metadata['video_upsmpl_idx_list']
        self.metadata['image_file_dict'] = image_file_dict
        
        self.metadata['recon_metadata'] = self.metadata['loc_metadata']
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Sampling done.')
        return 0


class UniformVideoSamplingTask(VideoSamplingTask):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs, prj_cfg)

        self.sampling_step = prj_cfg['sampling_config']['step']

    def sample_from_video(self, video_file_list) -> list:
        return video_file_list[::self.sampling_step]


class RigSamplingTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
    
        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
    
    @abstractmethod
    def sample(self) -> list:
        pass
        # return list of frame_idx to ln from img_upsmpl_dir to img_dir

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        image_upsmpl_file_dict = self.metadata['image_upsmpl_file_dict']
        image_file_dict = dict()
        image_upsmpl_file_mask_dict = None
        image_file_mask_dict = None
        if 'image_upsmpl_file_mask_dict' in self.metadata:
            image_upsmpl_file_mask_dict = self.metadata['image_upsmpl_file_mask_dict']
            image_file_mask_dict = dict()


        # clean img_dir, workaround
        shutil.rmtree(self.img_dir) 
        os.mkdir(self.img_dir)

        frame_idx_list = self.sample()
        for frame_idx in frame_idx_list:
            image_file_dict[frame_idx] = image_upsmpl_file_dict[frame_idx]
            [os.symlink(join(self.img_upsmpl_dir, filename), join(self.img_dir, filename)) for filename in image_file_dict[frame_idx]]
            if image_upsmpl_file_mask_dict is not None:
                image_file_mask_dict[frame_idx] = image_upsmpl_file_mask_dict[frame_idx]
                [os.symlink(join(self.img_upsmpl_dir, filename), join(self.img_dir, filename)) for filename in image_file_mask_dict[frame_idx]]

        self.metadata['frame_idx_list'] = frame_idx_list
        self.metadata['image_file_dict'] = image_file_dict
        if image_file_mask_dict is not None:
            self.metadata['image_file_mask_dict'] = image_file_mask_dict

        self.metadata['recon_metadata'] = self.metadata['loc_metadata']

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Sampling done.')
        return 0


class UniformRigSamplingTask(RigSamplingTask):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs, prj_cfg)

        self.sampling_step = prj_cfg['sampling_config']['step']

    def sample(self) -> list:
        sample_list = self.metadata['frame_upsmpl_idx_list'][self.sampling_step//2::self.sampling_step]

        return sample_list


class GoodFrameSelectionRigSamplingTask(RigSamplingTask):
    pass # TODO


class PersonMaskTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])

        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')

        self.stage_list = []
        self.stage_config_list = []

        stage_processing_handler = LoadImageStage()
        stage_config = {'base_path': self.img_dir}
        self.stage_list.append(
            {
                'name': 'load',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)

        person_mask_model_config = {
            'person_label': int(prj_cfg['person_mask_config']['person_label']), # 13,
            'MAX_INPUT_SIZE': prj_cfg['person_mask_config']['MAX_INPUT_SIZE'], # (257, 257),
            'MODEL_PATH': prj_cfg['person_mask_config']['MODEL_PATH']
        }
        self.person_mask_model = PersonMaskModel(person_mask_model_config)

        stage_processing_handler = PersonMaskPreprocStage(self.person_mask_model)
        stage_config = {}
        self.stage_list.append(
            {
                'name': 'preproc',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)

        stage_processing_handler = PersonMaskInferenceStage(self.person_mask_model)
        stage_config = {}
        self.stage_list.append(
            {
                'name': 'inference',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)

        stage_processing_handler = PersonMaskPostprocStage(self.person_mask_model)
        stage_config = {}
        self.stage_list.append(
            {
                'name': 'postproc',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)
        
        stage_processing_handler = SaveMaskStage()
        stage_config = {'base_path': self.img_dir}
        self.stage_list.append(
            {
                'name': 'save',
                'stage_processing_handler': stage_processing_handler,
                'BATCH_SIZE': 16, 
                'FEED_SIZE_LIMIT': 32
            }
        )
        self.stage_config_list.append(stage_config)

        assert len(self.stage_list) == len(self.stage_config_list)
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')
    
    
    def run_task_content(self):
        self.logger.info('Start running task content...')
        
        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        pipeline_config = {
            'FEED_SIZE_LIMIT': 32,
            'stage_config': self.stage_list
        }
        self.pipeline = Pipeline()
        self.pipeline.build_pipeline(pipeline_config)
        input_handler = FilenameInputHandler()
        self.pipeline.set_input_handler(input_handler)
        output_handler = NoOpOutputHandler()
        self.pipeline.set_output_handler(output_handler)
        input_handler.set({})
        for stage, stage_config in zip(self.stage_list, self.stage_config_list):
            stage['stage_processing_handler'].set(stage_config, self.logger, self.metadata)
        output_handler.set({})

        self.person_mask_model.start()
        self.pipeline.feed_sync([filename for l in self.metadata['image_file_dict'].values() for filename in l])
        image_file_dict, image_file_mask_dict, _ = get_image_file_dict(self.img_dir)
        self.metadata['image_file_mask_dict'] = image_file_mask_dict

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Image preprocessing done.')
        return 0


class SlicingTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
    
        self.slice_length = prj_cfg['slicing_config']['slice_length']
        self.overlap_length_list = prj_cfg['slicing_config']['overlap_length_list']# e.g. [[1, 3], [2, 3], [step, num to extend]]
        # e.g. slice_length = 5 overlap_length_list = [[1, 3], [2, 2]], if slice start from 5
        # =>  0 (pad 2 x2, but reach boundary)| 2, 3, 4 (pad 1 x3) | 5, 6, 7, 8, 9 | 10, 11, 12 (pad 1 x3)| 14, 16 (pad 2 x2) ||
        oshift = 0
        self.oshift_list = []
        for step, count in self.overlap_length_list:
            for _ in range(count):
                oshift += step
                self.oshift_list.append(oshift)

        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.slices_dir = join(prj_dirs['prj'], 'slices')
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        self.metadata['is_slicing'] = True

        image_upsmpl_file_dict = self.metadata['image_upsmpl_file_dict']
        image_file_dict = self.metadata['image_file_dict']
        image_upsmpl_file_mask_dict = None
        if 'image_upsmpl_file_mask_dict' in self.metadata:
            image_upsmpl_file_mask_dict = self.metadata['image_upsmpl_file_mask_dict']
        image_file_mask_dict = None
        if 'image_file_mask_dict' in self.metadata:
            image_file_mask_dict = self.metadata['image_file_mask_dict']
        frame_upsmpl_idx_list = self.metadata['frame_upsmpl_idx_list'] 
        loc_ss = self.metadata['loc_metadata']['ss'] # if 'ss' in self.metadata['loc_metadata'] else 0
        loc_to = self.metadata['loc_metadata']['to'] # if 'to' in self.metadata['loc_metadata'] else 
        loc_fps = self.metadata['loc_metadata']['fps']
        frame_idx_list = self.metadata['frame_idx_list'] 
        recon_ss = self.metadata['recon_metadata']['ss'] # if 'ss' in self.metadata['recon_metadata'] else 0
        recon_to = self.metadata['recon_metadata']['to'] # if 'to' in self.metadata['recon_metadata'] else 
        recon_fps = self.metadata['recon_metadata']['fps']
        slice_frame_upsmpl_idx_dict = dict()
        slice_beg_end_dict = dict()
        slice_dir_dict = dict()

        # clean slices_dir
        if os.path.exists(self.slices_dir):
            shutil.rmtree(self.slices_dir) 
        os.mkdir(self.slices_dir)

        slice_idx = 0
        beg = 0
        end = self.slice_length
        slice_beg = loc_ss
        upsmpl_i = 0
        flag = True
        while flag :
            slice_dir = join(self.slices_dir, '{:03}'.format(slice_idx))
            slice_img_upsmpl_dir = join(slice_dir, 'img_upsmpl')
            slice_img_dir = join(slice_dir, 'img')
            os.mkdir(slice_dir)
            os.mkdir(slice_img_upsmpl_dir)
            os.mkdir(slice_img_dir)
            slice_dir_dict[slice_idx] = slice_dir
            
            if end >= len(frame_idx_list): #last slice => align end
                flag = False
                end = len(frame_idx_list)
                beg = max(0, end - self.slice_length)
                slice_end = loc_to
                # find the 'upsmpl' frame idx belong to the slice and symlink the images 
                slice_frame_upsmpl_idx_dict[slice_idx] = frame_upsmpl_idx_list[upsmpl_i:]
                [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) 
                for frame_upsmpl_idx in frame_upsmpl_idx_list[upsmpl_i:] 
                for image_file in image_upsmpl_file_dict[frame_upsmpl_idx]]
                # if image_upsmpl_file_mask_dict is not None: # also symlink mask images if there are masks
                #     [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) 
                #     for frame_upsmpl_idx in frame_upsmpl_idx_list[upsmpl_i:] 
                #     for image_file in image_upsmpl_file_mask_dict[frame_upsmpl_idx]]
            else:
                # find the 'upsmpl' frame idx belong to the slice and symlink the images
                slice_end = frame_idx_list[end] / recon_fps + recon_ss
                slice_frame_upsmpl_idx_dict[slice_idx] = []
                while (frame_upsmpl_idx_list[upsmpl_i] / loc_fps + loc_ss) < slice_end:
                    slice_frame_upsmpl_idx_dict[slice_idx].append(frame_upsmpl_idx_list[upsmpl_i])
                    [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) for image_file in image_upsmpl_file_dict[frame_upsmpl_idx_list[upsmpl_i]]]
                    # if image_upsmpl_file_mask_dict is not None: # also symlink mask images if there are masks
                    #     [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) for image_file in image_upsmpl_file_mask_dict[frame_upsmpl_idx_list[upsmpl_i]]]
                    upsmpl_i += 1

            # ===================================================
            # find the 'reconstruction' frame idx belong to the slice and symlink the images
            # the main body of the slice
            [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) 
            for frame_idx in frame_idx_list[beg:end] 
            for image_file in image_file_dict[frame_idx]]
            if image_file_mask_dict is not None: # also symlink mask images if there are masks
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) 
                for frame_idx in frame_idx_list[beg:end] 
                for image_file in image_file_mask_dict[frame_idx]]

            # extend from beg
            for oshift in self.oshift_list:
                oi = beg - oshift
                if oi < 0:
                    break
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_dict[frame_idx_list[oi]]]
                if image_file_mask_dict is not None: # also symlink mask images if there are masks
                    [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_mask_dict[frame_idx_list[oi]]]
            
            # extend from end
            for oshift in self.oshift_list:
                oi = end - 1 + oshift
                if oi >= len(frame_idx_list):
                    break
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_dict[frame_idx_list[oi]]]
                if image_file_mask_dict is not None: # also symlink mask images if there are masks
                    [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_mask_dict[frame_idx_list[oi]]]

            slice_beg_end_dict[slice_idx] = (slice_beg, slice_end)
            slice_idx += 1
            beg = end
            end += self.slice_length
            slice_beg = slice_end

        self.metadata['slice_frame_upsmpl_idx_dict'] = slice_frame_upsmpl_idx_dict
        self.metadata['slice_beg_end_dict'] = slice_beg_end_dict
        self.metadata['slice_dir_dict'] = slice_dir_dict

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Slicing done.')
        return 0

class SlicingTask_w_keyIdx(Task):
    """
    Slice full image list  for independant reconstruction project
    In this version the key frames are given, and will be sliced into slices according to slice length.
    notice that between slices, there will be overlapped regions in order to eventually recover slices back to full path.
    This task will create folders of slices, as 000 001 002...etc.
    It will link the image files from original img and img_upsml folder into slicing folders.
    only frames in img folder will do reconstruction(key-frames)
    the freams in img_upsmpl folder will do localization to fill the full path.
    

    frame_idx_list: key frame index, should be corresponding to the image list in img folder. ex [1,2,5,9,15,...529,523,550]
    frame_upsmpl_idx_list: full frame index, should be corresponding to the image list in img_upsmpl folder. ex [0,1,2,3,.....,550]
    slice_length: number of key frames in each slice
    overlap_length_list: [[step, number to extend]]
        # e.g. slice_length = 5 overlap_length_list = [[1, 3], [2, 2]], consider a  slice starts from 5
        # =>  0 (pad 2 x2, but reach boundary)| 2, 3, 4 (pad 1 x3) || 5, 6, 7, 8, 9 || 10, 11, 12 (pad 1 x3)| 14, 16 (pad 2 x2) 
    

    example: key frame [0,1,5,9,10,17,21] full frames [0,1,2,3,4,....25]
             slice_length = 3
             overlap_length_list [1,2]
                       img: 0 1 5 | 9 10 17 | 10 17 21 <---last slice will be forced to have slice_length
              img extended: 0 1 5 9 10 | 1 5 9 10 17 21 | 5 9 10 17 21 (the extended frames is only used for connecting slices) 
    img_upsmpl: 0 1 2 3 4 5 6 | 7 8 9 10 11 12 13 14 15 16 17 18 | 10 11 12 13 14 15 16 17 18 19 20 21  
                extended area   original frames  extented area
    
    """
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
    
        self.slice_length = prj_cfg['slicing_config']['slice_length']
        self.overlap_length_list = prj_cfg['slicing_config']['overlap_length_list']# e.g. [[1, 3], [2, 3], [step, num to extend]]
        # e.g. slice_length = 5 overlap_length_list = [[1, 3], [2, 2]], if slice start from 5
        # =>  0 (pad 2 x2, but reach boundary)| 2, 3, 4 (pad 1 x3) | 5, 6, 7, 8, 9 | 10, 11, 12 (pad 1 x3)| 14, 16 (pad 2 x2) ||
        oshift = 0
        self.oshift_list = []
        for step, count in self.overlap_length_list:
            for _ in range(count):
                oshift += step
                self.oshift_list.append(oshift)

        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.img_dir = prj_dirs['img']
        self.prj_dir = prj_dirs['prj']
        self.slices_dir = join(prj_dirs['prj'], 'slices')
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')
        

        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
        self.metadata['is_slicing'] = True

        image_upsmpl_file_dict = self.metadata['image_upsmpl_file_dict']
        image_file_dict = self.metadata['image_file_dict']
        image_upsmpl_file_mask_dict = None
        if 'image_upsmpl_file_mask_dict' in self.metadata:
            image_upsmpl_file_mask_dict = self.metadata['image_upsmpl_file_mask_dict']
        image_file_mask_dict = None
        if 'image_file_mask_dict' in self.metadata:
            image_file_mask_dict = self.metadata['image_file_mask_dict']
        frame_upsmpl_idx_list = self.metadata['frame_upsmpl_idx_list'] 
        frame_idx_list = self.metadata['frame_idx_list'] 
        slice_frame_upsmpl_idx_dict = dict()
        slice_beg_end_dict = dict()
        slice_dir_dict = dict()

        # clean slices_dir
        if os.path.exists(self.slices_dir):
            shutil.rmtree(self.slices_dir) 
        os.mkdir(self.slices_dir)

        slice_idx = 0
        beg = 0
        end = self.slice_length
        slice_beg = 0
        upsmpl_i = 0
        flag = True
        while flag :
            slice_dir = join(self.slices_dir, '{:03}'.format(slice_idx))
            slice_img_upsmpl_dir = join(slice_dir, 'img_upsmpl')
            slice_img_dir = join(slice_dir, 'img')
            os.mkdir(slice_dir)
            os.mkdir(slice_img_upsmpl_dir)
            os.mkdir(slice_img_dir)
            slice_dir_dict[slice_idx] = slice_dir
            
            if end >= len(frame_idx_list): #last slice => align end
                flag = False
                end = len(frame_idx_list)
                beg = max(0, end - self.slice_length)
                slice_end = 1 
                # find the 'upsmpl' frame idx belong to the slice and symlink the images 
                slice_frame_upsmpl_idx_dict[slice_idx] = frame_upsmpl_idx_list[upsmpl_i:]
                [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) 
                for frame_upsmpl_idx in frame_upsmpl_idx_list[upsmpl_i:] 
                for image_file in image_upsmpl_file_dict[frame_upsmpl_idx]]
            else:
                # find the 'upsmpl' frame idx belong to the slice and symlink the images
                slice_end = (frame_idx_list[end] + frame_idx_list[end+1]) / 2 
                slice_end = slice_end / len(frame_upsmpl_idx_list) # this is now the fraction of total length
                slice_frame_upsmpl_idx_dict[slice_idx] = []
                while (frame_upsmpl_idx_list[upsmpl_i] / len(frame_upsmpl_idx_list)) < slice_end:
                    slice_frame_upsmpl_idx_dict[slice_idx].append(frame_upsmpl_idx_list[upsmpl_i])
                    [os.symlink(join(self.img_upsmpl_dir, image_file), join(slice_img_upsmpl_dir, image_file)) for image_file in image_upsmpl_file_dict[frame_upsmpl_idx_list[upsmpl_i]]]
                    upsmpl_i += 1

            # ===================================================
            # find the 'reconstruction' frame idx belong to the slice and symlink the images
            # the main body of the slice
            [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) 
            for frame_idx in frame_idx_list[beg:end] 
            for image_file in image_file_dict[frame_idx]]
            if image_file_mask_dict is not None: # also symlink mask images if there are masks
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) 
                for frame_idx in frame_idx_list[beg:end] 
                for image_file in image_file_mask_dict[frame_idx]]

            # extend from beg
            for oshift in self.oshift_list:
                oi = beg - oshift
                if oi < 0:
                    break
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_dict[frame_idx_list[oi]]]
                if image_file_mask_dict is not None: # also symlink mask images if there are masks
                    [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_mask_dict[frame_idx_list[oi]]]
            
            # extend from end
            for oshift in self.oshift_list:
                oi = end - 1 + oshift
                if oi >= len(frame_idx_list):
                    break
                [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_dict[frame_idx_list[oi]]]
                if image_file_mask_dict is not None: # also symlink mask images if there are masks
                    [os.symlink(join(self.img_dir, image_file), join(slice_img_dir, image_file)) for image_file in image_file_mask_dict[frame_idx_list[oi]]]

            slice_beg_end_dict[slice_idx] = (slice_beg, slice_end)
            slice_idx += 1
            beg = end
            end += self.slice_length
            slice_beg = slice_end

        self.metadata['slice_frame_upsmpl_idx_dict'] = slice_frame_upsmpl_idx_dict
        self.metadata['slice_beg_end_dict'] = slice_beg_end_dict
        self.metadata['slice_dir_dict'] = slice_dir_dict

        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        self.logger.info('Slicing done.')
        return 0


class InitReconLocTask(Task):
    def __init__(self, step: int, prj_dirs: dict, prj_cfg: dict):
        ''' Get config parameters & directories needed, and do some checks. '''
        
        super().__init__(step, prj_dirs['log'])
    
        self.recon_config_template = prj_cfg['recon_config_template']
        if 'loc_config_template' in prj_cfg:
            self.loc_config_template = prj_cfg['loc_config_template']
        self.is_scene = self.recon_config_template['is_scene']

        self.prj_dir = prj_dirs['prj']
        self.img_upsmpl_dir = prj_dirs['img_upsmpl']
        self.metadata_file = join(prj_dirs['prj'], 'metadata.pkl')
        
        # Print the members of the object
        self.logger.info(f'Object: {self.__dict__}')

    def gen_route_scene(self, video, img_upsmpl_dir, slice_idx, frame_upsmpl_idx_list, ss, to, fps, slice_beg_timestamp, slice_end_timestamp):
        route = dict()
        route['is_scene'] = True
        route['video'] = video
        route['slice_beg_timestamp'] = slice_beg_timestamp
        route['slice_end_timestamp'] = slice_end_timestamp
        route['frames'] = []
        for frame_upsmpl_idx in frame_upsmpl_idx_list:
            n_lenses = len(self.metadata['image_upsmpl_file_dict'][frame_upsmpl_idx])
            route['frames'].append(
                {
                    'dir': img_upsmpl_dir, 
                    'slice': slice_idx, 
                    'frame_idx': frame_upsmpl_idx,
                    'timestamp': frame_upsmpl_idx / fps + ss, # frame_upsmpl_idx start from 0 !!
                    'filenames': self.metadata['image_upsmpl_file_dict'][frame_upsmpl_idx],
                    'lense_positions': [None] * n_lenses,
                    'lense_rotations': [None] * n_lenses,
                    'lense_footprints': [None] * n_lenses,
                    'position': None,
                    'rotation': None,
                    'footprint': None
                }
            )

        return route

    def gen_route_obj(self, img_upsmpl_dir):
        route = dict()
        route['is_scene'] = False
        route['dir'] = img_upsmpl_dir
        route['position'] = None
        route['rotation'] = None
            
        return route

    def save_route(self, base_dir, route):
        route['last_overwritten'] = get_current_time_str()
        with open(join(base_dir, 'routes.json'), 'w') as f:
            safe_json_dump(route, f)

    def gen_recon_config(self, img_prj_dir, slice_idx=-1):
        recon_config = copy.deepcopy(self.recon_config_template)
        if slice_idx >= 0:
            recon_config['project']['project_dir'] = join(recon_config['project']['project_dir'], '{:03}'.format(slice_idx))
            recon_config['project']['name'] = recon_config['project']['name'] + '_{:03}'.format(slice_idx)
        recon_config['images']['img_prj_dir'] = img_prj_dir
        recon_config['sfm']['camera_model'] = self.metadata['recon_metadata']['camera_model']
        if recon_config['sfm']['camera_model'] == 7:
            recon_config['sfm']['focal_length'] = -1
        else:
            recon_config['sfm']['focal_length'] = self.metadata['recon_metadata']['focal_length']

        if 'loc_metadata' in self.metadata:
            recon_config['sfm']['loc_camera_model'] = self.metadata['loc_metadata']['camera_model']
            if recon_config['sfm']['loc_camera_model'] == 7:
                recon_config['sfm']['loc_focal_length'] = -1
            else:
                recon_config['sfm']['loc_focal_length'] = self.metadata['loc_metadata']['focal_length']

        return recon_config

    def gen_loc_config(self, recon_dir, slice_idx=-1) -> dict:
        if slice_idx <= 0:
            return
        else:
            loc_config = copy.deepcopy(self.loc_config_template)
            if slice_idx == 1:
                loc_config['transform']['prj_dir_main'] = join(recon_dir, '{:03}'.format(slice_idx - 1))
            else:
                loc_config['transform']['prj_dir_main'] = loc_config['project']['project_dir']
            loc_config['transform']['prj_dir_sub'] = join(recon_dir, '{:03}'.format(slice_idx))
            loc_config['transform']['data_dir_name_main'] = 'data_' + loc_config['project']['name'] + '_{:03}'.format(slice_idx - 1)
            loc_config['transform']['data_dir_name_sub'] = 'data_' + loc_config['project']['name'] + '_{:03}'.format(slice_idx)
        
        return loc_config

    def save_recon_config(self, base_dir, recon_config):
        file_path = join(base_dir, 'recon_config.yaml')
        safe_yaml_dump(recon_config, file_path)

    def save_loc_config(self, base_dir, loc_config):
        file_path = join(base_dir, 'loc_config.yaml')
        safe_yaml_dump(loc_config, file_path)

    def run_task_content(self):
        self.logger.info('Start running task content...')

        with open(self.metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        is_slicing = False
        if 'is_slicing' in self.metadata:
            is_slicing = self.metadata['is_slicing']

        if is_slicing:
            for slice_idx in self.metadata['slice_frame_upsmpl_idx_dict']:
                slice_dir = self.metadata['slice_dir_dict'][slice_idx]
                img_upsmpl_dir = join(slice_dir, 'img_upsmpl')
                if self.is_scene:
                    frame_upsmpl_idx_list = self.metadata['slice_frame_upsmpl_idx_dict'][slice_idx]
                    slice_beg_timestamp, slice_end_timestamp = self.metadata['slice_beg_end_dict'][slice_idx]
                    route = self.gen_route_scene(self.metadata['loc_metadata']['video'], img_upsmpl_dir, slice_idx, frame_upsmpl_idx_list, float(self.metadata['loc_metadata']['ss']), float(self.metadata['loc_metadata']['to']), float(self.metadata['loc_metadata']['fps']), slice_beg_timestamp, slice_end_timestamp)
                else:
                    route = self.gen_route_obj(img_upsmpl_dir)
                self.save_route(slice_dir, route)
                recon_config = self.gen_recon_config(slice_dir, slice_idx)
                self.save_recon_config(slice_dir, recon_config)
                loc_config = self.gen_loc_config(self.recon_config_template['project']['project_dir'], slice_idx)
                self.save_loc_config(slice_dir, loc_config)
        else:
            if self.is_scene:
                frame_upsmpl_idx_list = self.metadata['frame_upsmpl_idx_list'] 
                route = self.gen_route_scene(self.metadata['loc_metadata']['video'], self.img_upsmpl_dir, 0, frame_upsmpl_idx_list, float(self.metadata['loc_metadata']['ss']), float(self.metadata['loc_metadata']['to']), float(self.metadata['loc_metadata']['fps']), float(self.metadata['loc_metadata']['ss']), float(self.metadata['loc_metadata']['to']))
            else:
                route = self.gen_route_obj(self.img_upsmpl_dir)
            self.save_route(self.prj_dir, route)
            recon_config = self.gen_recon_config(self.prj_dir)
            self.save_recon_config(self.prj_dir, recon_config)
            
        self.logger.info('Initialization for reconstruction & localization projects done.')
        return 0


