import os
from multiprocessing.dummy import Pool as ThreadPool
from PIL import Image

from processor.pipeline import StageProcessingHandler
from .person_mask_model import PersonMaskModel


class PersonMaskPreprocStage(StageProcessingHandler):
    def __init__(self, person_mask_model: PersonMaskModel):
        self.person_mask_model = person_mask_model

    def set(self, config: dict, logger, metadata: dict):
        pass

    def process(self, batch):
        frame_list = [frame_info['image'] for frame_info in batch]
        pad_frame_list, valid_hw_list = self.person_mask_model.preproc(frame_list)
        assert len(pad_frame_list) == len(frame_list)
        assert len(valid_hw_list) == len(frame_list)

        for frame_info, pad_frame, valid_hw in zip(batch, pad_frame_list, valid_hw_list):
            frame_info['pad_frame'] = pad_frame
            frame_info['valid_hw'] = valid_hw
            
        return batch


class PersonMaskInferenceStage(StageProcessingHandler):
    def __init__(self, person_mask_model: PersonMaskModel):
        self.person_mask_model = person_mask_model

    def set(self, config: dict, logger, metadata: dict):
        pass

    def process(self, batch):
        pad_frame_list = [frame_info['pad_frame'] for frame_info in batch]
        mask_list = self.person_mask_model.inference(pad_frame_list)
        assert len(mask_list) == len(pad_frame_list)

        for frame_info, mask in zip(batch, mask_list):
            frame_info['mask'] = mask
            del frame_info['pad_frame']

        return batch


class PersonMaskPostprocStage(StageProcessingHandler):
    def __init__(self, person_mask_model: PersonMaskModel):
        self.person_mask_model = person_mask_model

    def set(self, config: dict, logger, metadata: dict):
        pass

    def process(self, batch):
        frame_list = [frame_info['image'] for frame_info in batch]
        valid_hw_list = [frame_info['valid_hw'] for frame_info in batch]
        mask_list = [frame_info['mask'] for frame_info in batch]

        mask_list = self.person_mask_model.postproc(frame_list, valid_hw_list, mask_list)

        for frame_info, mask in zip(batch, mask_list):
            frame_info['mask'] = mask
            del frame_info['valid_hw']

        return batch


class SaveMaskStage(StageProcessingHandler):
    def __init__(self):
        self.base_path = None # ''
        self.num_cpu = None # 32
        self.pool = None # ThreadPool(processes=self.num_cpu)
        self.logger = None
        self.save_image_function = self.save_mask

    def set(self, config: dict, logger, metadata: dict):
        self.base_path = config['base_path'] 
        self.num_cpu = 4
        if 'num_cpu' in config:
            self.num_cpu = int(config['num_cpu'])
        self.pool = ThreadPool(processes=self.num_cpu)
        self.logger = logger

    def save_mask(self, image_info):
        try:
            filename = '{}_mask.png'.format(os.path.splitext(image_info['filename'])[0])
            img_path = os.path.join(self.base_path, filename)
            Image.fromarray(image_info['mask']).save(img_path)

            return image_info
                
        except:
            if self.logger is not None:
                self.logger.info('bad file: {}\n'.format(img_path.encode('utf-8')))

            return None

    def process(self, batch):
        image_info_list = self.pool.map(self.save_image_function, batch)
        image_info_list = [image_info for image_info in image_info_list if image_info is not None]

        return image_info_list

        