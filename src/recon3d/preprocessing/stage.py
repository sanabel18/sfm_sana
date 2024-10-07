from abc import ABC, abstractmethod
import os, sys, fnmatch
import json
import numpy as np
import cv2
from PIL import Image
import shutil
import copy
from itertools import chain

from videoio.video_reader import VideoReader
from videoio.video_info import VideoInfo
from videoio.converter import Converter
from videoio.reader_option import ReaderOption
from utils360.cubemap_converter import CubemapConverter
from utils360.converter_option import ConverterOption

from multiprocessing.dummy import Pool as ThreadPool
from processor.pipeline import StageProcessingHandler


class MyVideoReader(VideoReader): # override init
    """
    It is a wrapper of ffmpeg to read video frames via pipe.
    """

    def __init__(
            self,
            file_path,
            try_gpu=False,
            ss=None, to=None, fps=None
        ):
        """
        Args:
            file_path: str. input video path
            try_gpu: boolean. try to use gpu decoder
        """

        self.video_path = file_path
        self.video_info_obj = VideoInfo(file_path)
        self.reader = None

        self.to_bgr_converter = None # should be converter

        global_opt, input_opt, output_opt = MyVideoReader.set_ffmpeg_opt(ss, to, fps)
        self.reader_option = ReaderOption(
            try_gpu,
            self.video_info_obj,
            global_opt,
            input_opt,
            output_opt
        )

        self.width, self.height = self.video_info_obj.get_scale()
        self.rotate = '0'
        if 'rotate' in self.video_info_obj.video_stream['tags']:
            self.rotate = self.video_info_obj.video_stream['tags']['rotate']
        if self.rotate == '0':
            pass
        elif self.rotate == '90':
            self.width, self.height = self.height, self.width
        elif self.rotate == '180':
            pass
        elif self.rotate == '270':
            self.width, self.height = self.height, self.width

        input_pix_fmt = self.video_info_obj.get_pix_fmt()
        output_pix_fmt = output_opt['pix_fmt']
        if output_pix_fmt == 'rgb24' or output_pix_fmt == 'bgr24':
            if input_pix_fmt == output_pix_fmt:
                pass
            elif input_pix_fmt == 'yuv444p':
                pass
            else:
                if input_pix_fmt != 'yuv420p':
                    # logger.warning("Input pixel format is not tested. The result may not be lossless.")
                    pass

                # change reader output pix_fmt to yuv444p and set converter
                # logger.info("Create a converter, read as yuv444p first and convert to {} later.".format(output_pix_fmt))

                self.reader_option.update_option('output', 'pix_fmt', 'yuv444p')
                self.to_bgr_converter = Converter('yuv444p', output_pix_fmt, self.width, self.height)

        self.cmd = self.reader_option.compile_cmd(self.video_path, "pipe:")


    @staticmethod
    def set_ffmpeg_opt(ss=None, to=None, fps=None):
        global_opt = dict()
        input_opt = dict()
        output_opt = dict()
        if ss is not None:
            output_opt['ss'] = str(ss)
        if to is not None:
            output_opt['to'] = str(to)
        # '-qscale:v' '2'
        if fps is not None:
            output_opt['vf'] = 'fps={}'.format(fps)
        output_opt['f'] = 'rawvideo'
        output_opt['pix_fmt'] = 'rgb24'

        return global_opt, input_opt, output_opt


def set_image_info(filename, image):
    image_info = {
        'filename': filename,
        'image': image
    }

    return image_info


class SourceGenerator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        self.config = config
        self.logger = logger
        self.metadata = metadata
        if 'camera_model' in self.config:
            self.metadata['camera_model'] = self.config['camera_model']
        if 'focal_length' in self.config:
            self.metadata['focal_length'] = self.config['focal_length']

    @abstractmethod
    def gen(self):
        pass


class ImageDirSourceGenerator(SourceGenerator):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_dir = self.config['src_dir']
        self.metadata['src_type'] = 'image'
        
    def gen(self):
        for filename in os.listdir(self.src_dir):
            try:
                img_path = os.path.join(self.src_dir, filename)
                image = np.array(Image.open(img_path).convert('RGB'))
                yield set_image_info(filename, image)
                    
            except:
                self.logger.info('bad file: {}\n'.format(img_path.encode('utf-8')))

                continue


class ObjVideoSourceGenerator(SourceGenerator):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_video_list = self.config['src_video_list']
        self.ss_list = self.config['ss_list']
        self.to_list = self.config['to_list']
        self.fps_list = self.config['fps_list']

        assert len(self.src_video_list) == len(self.ss_list)
        assert len(self.src_video_list) == len(self.to_list)
        assert len(self.src_video_list) == len(self.fps_list)

        self.metadata['src_type'] = 'obj_video_list'
        self.metadata['video'] = self.src_video_list

    def gen(self):
        for vid, (src_video, ss, to, fps) in enumerate(zip(self.src_video_list, self.ss_list, self.to_list, self.fps_list)):
            with MyVideoReader(file_path=src_video, try_gpu=False, ss=ss, to=to, fps=fps) as reader:
                if fps is None:
                    fps = reader.video_info_obj.get_framerate()

                frame_idx = 0
                while True:
                    ret, frame = reader.read()
                    if not ret:
                        break
                    filename = '{}_{:05d}.jpg'.format(vid, frame_idx)
                    frame_idx += 1
                    yield set_image_info(filename, frame)


class SingleVideoSourceGenerator(SourceGenerator):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_video = config['src_video']
        self.ss = None 
        if 'ss' in self.config:
            self.ss = self.config['ss']
        self.to = None 
        if 'to' in self.config:
            self.to = self.config['to']
        self.fps = None 
        if 'fps' in self.config:
            self.fps = self.config['fps']

        self.metadata['src_type'] = 'video'
        self.metadata['video'] = self.src_video
        self.metadata['ss'] = self.ss
        self.metadata['to'] = self.to

    def gen(self):
        with MyVideoReader(file_path=self.src_video, try_gpu=False, ss=self.ss, to=self.to, fps=self.fps) as reader:
            if self.fps is None:
                self.metadata['fps'] = reader.video_info_obj.get_framerate()
            else:
                self.metadata['fps'] = self.fps

            frame_idx = 0
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                filename = '{:05d}.jpg'.format(frame_idx)
                frame_idx += 1
                yield set_image_info(filename, frame)


class Insta360SourceGenerator(SourceGenerator):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_dir = self.config['src_dir']
        self.ss = None 
        if 'ss' in self.config:
            self.ss = self.config['ss']
        self.to = None 
        if 'to' in self.config:
            self.to = self.config['to']
        self.fps = None 
        if 'fps' in self.config:
            self.fps = self.config['fps']

        self.metadata['src_type'] = 'insta360'
        self.metadata['video'] = self.src_dir
        # self.metadata['camera_model'] = 5
        self.metadata['ss'] = self.ss
        self.metadata['to'] = self.to
    
    def gen(self):
        for cam_id in range(1, 7):
            src_video = os.path.join(self.src_dir, 'origin_{}.mp4'.format(cam_id))
            with MyVideoReader(file_path=src_video, try_gpu=False, ss=self.ss, to=self.to, fps=self.fps) as reader:
                if self.fps is None:
                    self.metadata['fps'] = reader.video_info_obj.get_framerate()
                else:
                    self.metadata['fps'] = self.fps

                frame_idx = 0
                while True:
                    ret, frame = reader.read()
                    if not ret:
                        break
                    filename = '{:05d}_{}.jpg'.format(frame_idx, cam_id)
                    frame_idx += 1
                    yield set_image_info(filename, frame)


class EquirectSourceGenerator_w_keyIdx(SourceGenerator):
    """
    parse frames from video, only return the frame info
    if the frame index is within key frame index list.
    """
    def __init__(self):
        pass

    @abstractmethod
    def search_video(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_dir = self.config['src_dir']
        self.video_path = self.config['video_path']
        self.ss = None 
        if 'ss' in self.config:
            self.ss = self.config['ss']
        self.to = None 
        if 'to' in self.config:
            self.to = self.config['to']

        self.metadata['src_type'] = 'equirect'
        self.metadata['src_dir'] = self.src_dir
        self.metadata['video'] = self.video_path
        self.metadata['camera_model'] = 7
        self.metadata['focal_length'] = -1
        self.metadata['ss'] = self.ss
        self.metadata['to'] = self.to
        key_frame_np_file = self.config['key_frame_idx_file']
        self.metadata['key_frame_idx_list'] =  sorted(np.load(key_frame_np_file).astype(int).tolist())
 
        self.key_frame_set = set(np.load(key_frame_np_file).astype(int))

    
    def gen(self):
        with MyVideoReader(file_path=self.video_path, try_gpu=False, ss=self.ss, to=self.to) as reader:
            frame_idx = -1
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx in self.key_frame_set: 
                    filename = '{:05d}.jpg'.format(frame_idx)
                    yield set_image_info(filename, frame)




class EquirectSourceGenerator(SourceGenerator):
    def __init__(self):
        pass

    @abstractmethod
    def search_video(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        super().set(config, logger, metadata)
        self.src_dir = self.config['src_dir']
        self.video_path = self.config['video_path']

        self.ss = None 
        if 'ss' in self.config:
            self.ss = self.config['ss']
        self.to = None 
        if 'to' in self.config:
            self.to = self.config['to']
        self.fps = None 
        if 'fps' in self.config:
            self.fps = self.config['fps']

        self.metadata['src_type'] = 'equirect'
        self.metadata['src_dir'] = self.src_dir
        self.metadata['video'] = self.video_path
        self.metadata['camera_model'] = 7
        self.metadata['focal_length'] = -1
        self.metadata['ss'] = self.ss
        self.metadata['to'] = self.to
    
    def gen(self):
        with MyVideoReader(file_path=self.video_path, try_gpu=False, ss=self.ss, to=self.to, fps=self.fps) as reader:
            if self.fps is None:
                self.metadata['fps'] = reader.video_info_obj.get_framerate()
            else:
                self.metadata['fps'] = self.fps

            frame_idx = 0
            while True:
                ret, frame = reader.read()
                if not ret:
                    break
                filename = '{:05d}.jpg'.format(frame_idx)
                frame_idx += 1
                yield set_image_info(filename, frame)

# TODO:
# search_video should be integrated with tower api that find file by label !!!!
class Insta360EquirectSourceGenerator(EquirectSourceGenerator):
    def search_video(self):
        # Look for stitched.mp4, if there is no such video then look for VID
        #video_name_candidate = fnmatch.filter(os.listdir(self.src_dir), 'stitched*.mp4')
        video_name_candidate = fnmatch.filter(os.listdir(self.src_dir), '*stitch*.mp4')
        if len(video_name_candidate) > 0:
            return os.path.join(self.src_dir, video_name_candidate[0])
        video_name_candidate = fnmatch.filter(os.listdir(self.src_dir), 'VID_*.mp4')
        if len(video_name_candidate) > 0:
            return os.path.join(self.src_dir, video_name_candidate[0])
        raise FileNotFoundError


class GoProEquirectSourceGenerator(EquirectSourceGenerator):
    def search_video(self):
        # Look for stitched.mp4, if there is no such video then look for VID
        video_name_candidate = fnmatch.filter(os.listdir(self.src_dir), '*.mp4')
        if len(video_name_candidate) > 0:
            return os.path.join(self.src_dir, video_name_candidate[0])
        raise FileNotFoundError

class GoProEquirectSourceGenerator_w_keyIdx(EquirectSourceGenerator_w_keyIdx):
    def search_video(self):
        # Look for stitched.mp4, if there is no such video then look for VID
        video_name_candidate = fnmatch.filter(os.listdir(self.src_dir), '*.mp4')
        if len(video_name_candidate) > 0:
            return os.path.join(self.src_dir, video_name_candidate[0])
        raise FileNotFoundError



class RotateStage(StageProcessingHandler):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)

    def rotate(self, image_info):
        # rotate ccw
        img = image_info['image']
        img = cv2.transpose(img)
        img = cv2.flip(img,flipCode=0)
        image_info['image'] = img

        return image_info

    def process(self, batch):
        batch = self.pool.map(self.rotate, batch)

        return batch


class DefisheyeStage(StageProcessingHandler):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        param = np.load(config['k_d_npz'], mmap_mode='r')
        self.DIM = tuple(param['DIM'])
        self.K = param['K']
        self.D = param['D']
        self.Knew = self.K.copy()
        scale = 0.8
        self.Knew[(0, 1), (0, 1)] = scale * self.Knew[(0, 1), (0, 1)]
        metadata['camera_model'] = 3 # maybe 1 is ok
        metadata['focal_length'] = float(self.Knew[0][0])
        metadata['defisheye'] = True
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def defisheye(self, image_info):
        img = image_info['image']
        img = cv2.fisheye.undistortImage(img, self.K, self.D, Knew=self.Knew, new_size=self.DIM)
        image_info['image'] = img

        return image_info

    def process(self, batch):
        batch = self.pool.map(self.defisheye, batch)

        return batch


class DownsizeStage(StageProcessingHandler):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        self.power_factor = int(config['power_factor']) # downsize length to 1/2^power_factor
        metadata['focal_length'] = metadata['focal_length'] / pow(2, self.power_factor)
        metadata['downsize_power_factor'] = self.power_factor
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def downsize(self, image_info):
        img = image_info['image']
        h, w, c = img.shape
        img = cv2.resize(img, (w>>self.power_factor, h>>self.power_factor), interpolation=cv2.INTER_AREA)
        image_info['image'] = img

        return image_info

    def process(self, batch):
        batch = self.pool.map(self.downsize, batch)

        return batch


class CubemapStage(StageProcessingHandler):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        frame_size = 1024
        if 'frame_size' in config:
            frame_size = config['frame_size']
        converter_options = ConverterOption()
        self.cubemapConverter = CubemapConverter(converter_options, frame_size)
        metadata['camera_model'] = 3 # maybe 1 is ok
        metadata['focal_length'] = frame_size / 2.0
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def cubemap(self, image_info):
        image = image_info['image']
        cubemap = self.cubemapConverter.convert(image)
        order = [4, 5, 1, 0] # front, back, left, right
        image_info_list = []
        for i in range(len(order)):
            image_info_copy = copy.deepcopy(image_info)
            image_info_copy['image'] = cubemap[order[i]]
            image_info_copy['filename'] = '{}_{}.jpg'.format(os.path.splitext(image_info['filename'])[0], i)
            image_info_list.append(image_info_copy)

        return image_info_list

    def process(self, batch):
        batch = list(chain.from_iterable(self.pool.map(self.cubemap, batch)))

        return batch


# class DenseCubemapStage(StageProcessingHandler):
#     def __init__(self):
#         pass

#     def set(self, config: dict, logger, metadata: dict):
#         frame_size = 1024
#         if 'frame_size' in config:
#             frame_size = config['frame_size']
#         converter_options_list = [ConverterOption(), ConverterOption(rotate_x=-30, rotate_y=45), ConverterOption(rotate_x=-30, rotate_y=135), ConverterOption(rotate_x=-30, rotate_y=225), ConverterOption(rotate_x=-30, rotate_y=315)]
#         self.cubemapConverter_list = [CubemapConverter(converter_options, frame_size) for converter_options in converter_options_list]
#         self.order_list = [[4, 5, 1, 0], [4, 5], [4, 5], [4, 5], [4, 5]]
#         metadata['camera_model'] = 3 # maybe 1 is ok
#         metadata['focal_length'] = frame_size / 2.0
#         self.num_cpu = 4
#         if 'num_cpu' in config:
#             self.num_cpu = int(config['num_cpu'])
#         self.pool = ThreadPool(processes=self.num_cpu)
#         self.logger = logger

#     def cubemap(self, image_info):
#         image = image_info['image']
#         image_info_list = []
#         i = 0
#         for cubemapConverter, order in zip(self.cubemapConverter_list, self.order_list):
#             cubemap = cubemapConverter.convert(image)
        
#             for o in order:
#                 image_info_copy = copy.deepcopy(image_info)
#                 image_info_copy['image'] = cubemap[o]
#                 image_info_copy['filename'] = '{}_{}.jpg'.format(os.path.splitext(image_info['filename'])[0], i)
#                 image_info_list.append(image_info_copy)
#                 i += 1

#         return image_info_list

#     def process(self, batch):
#         batch = list(chain.from_iterable(self.pool.map(self.cubemap, batch)))

#         return batch

class DenseCubemapStage(StageProcessingHandler):
    def __init__(self):
        pass

    def set(self, config: dict, logger, metadata: dict):
        frame_size = 1024
        if 'frame_size' in config:
            frame_size = config['frame_size']
        converter_options_list = [ConverterOption(), ConverterOption(rotate_y=45)]
        self.cubemapConverter_list = [CubemapConverter(converter_options, frame_size) for converter_options in converter_options_list]
        self.order_list = [[4, 0, 5, 1], [4, 0, 5, 1]]
        self.rig_order = [0, 4, 1, 5, 2, 6, 3, 7]
        metadata['camera_model'] = 3 # maybe 1 is ok
        metadata['focal_length'] = frame_size / 2.0
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def cubemap(self, image_info):
        image = image_info['image']
        image_info_list = []
        i = 0
        for cubemapConverter, order in zip(self.cubemapConverter_list, self.order_list):
            cubemap = cubemapConverter.convert(image)
        
            for o in order:
                image_info_copy = copy.deepcopy(image_info)
                image_info_copy['image'] = cubemap[o]
                image_info_copy['filename'] = '{}_{}.jpg'.format(os.path.splitext(image_info['filename'])[0], self.rig_order[i])
                image_info_list.append(image_info_copy)
                i += 1

        return image_info_list

    def process(self, batch):
        batch = list(chain.from_iterable(self.pool.map(self.cubemap, batch)))

        return batch



class LoadImageStage(StageProcessingHandler):
    def __init__(self):
        self.base_path = None # ''
        self.num_cpu = None # 32
        self.pool = None # ThreadPool(processes=self.num_cpu)
        self.logger = None
        self.load_image_function = self.load_image

    def set(self, config: dict, logger, metadata: dict):
        self.base_path = config['base_path'] 
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def load_image(self, image_info):
        try:
            img_path = os.path.join(self.base_path, image_info['filename'])
            image = np.array(Image.open(img_path).convert('RGB'))
            image_info['image'] = image

            return image_info
                
        except:
            if self.logger is not None:
                self.logger.info('bad file: {}\n'.format(img_path.encode('utf-8')))

            return None

    def process(self, batch):
        image_info_list = self.pool.map(self.load_image_function, batch)
        image_info_list = [image_info for image_info in image_info_list if image_info is not None]

        return image_info_list


class ImageHashFilenameStage(StageProcessingHandler):
    def __init__(self):
        self.num_cpu = None # 32
        self.pool = None # ThreadPool(processes=self.num_cpu)
        self.logger = None
        self.image_hash_filename_function = self.image_hash_filename

    def set(self, config: dict, logger, metadata: dict):
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def image_hash_filename(self, image_info):
        basename, extension = os.path.splitext(image_info['filename'])
        image_hash = abs(hash(image_info['image'].data.tobytes()))
        image_info['filename'] = '{}_{}{}'.format(basename, image_hash, extension)
        
        return image_info
                

    def process(self, batch):
        image_info_list = self.pool.map(self.image_hash_filename_function, batch)

        return image_info_list


class SaveImageStage(StageProcessingHandler):
    def __init__(self):
        self.base_path = None # ''
        self.num_cpu = None # 32
        self.pool = None # ThreadPool(processes=self.num_cpu)
        self.logger = None
        self.save_image_function = self.save_image

    def set(self, config: dict, logger, metadata: dict):
        self.base_path = config['base_path'] 
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def save_image(self, image_info):
        try:
            img_path = os.path.join(self.base_path, image_info['filename'])
            if image_info['filename'].lower().endswith('.jpg') or image_info['filename'].lower().endswith('.jpeg'):
                Image.fromarray(image_info['image']).save(img_path, format='JPEG', subsampling=0, quality=100)
            else:
                Image.fromarray(image_info['image']).save(img_path)

            return image_info
                
        except:
            if self.logger is not None:
                self.logger.info('bad file: {}\n'.format(img_path.encode('utf-8')))

            return None

    def process(self, batch):
        image_info_list = self.pool.map(self.save_image_function, batch)
        image_info_list = [image_info for image_info in image_info_list if image_info is not None]

        return image_info_list
