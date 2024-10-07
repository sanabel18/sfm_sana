import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processor.pipeline import Pipeline
from processor.stage import FilenameInputHandler, LoadImageStage, NoOpOutputHandler
from person_mask_model import PersonMaskModel
from person_mask_stage import PersonMaskPreprocStage, PersonMaskInferenceStage, PersonMaskPostprocStage, SaveMaskStage







if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    image_dir = sys.argv[2]
    output_dir = sys.argv[3]


    load_stage = LoadImageStage()
    person_mask_model_config = {
        'person_label': 13, #15, #11, #13,
        'MAX_INPUT_SIZE': (257, 257),
        'MODEL_PATH': MODEL_PATH
    }
    person_mask_model = PersonMaskModel(person_mask_model_config)
    person_mask_model.start()
    preproc_stage = PersonMaskPreprocStage(person_mask_model)
    inference_stage = PersonMaskInferenceStage(person_mask_model)
    postproc_stage = PersonMaskPostprocStage(person_mask_model)
    save_stage = SaveMaskStage()
    
    pipeline_config = {
        'FEED_SIZE_LIMIT': 32,
        'stage_config': [
            {'name': 'load', 'stage_processing_handler': load_stage, 'BATCH_SIZE': 16, 'FEED_SIZE_LIMIT': 32},
            {'name': 'prerpoc', 'stage_processing_handler': preproc_stage, 'BATCH_SIZE': 16, 'FEED_SIZE_LIMIT': 32},
            {'name': 'inference', 'stage_processing_handler': inference_stage, 'BATCH_SIZE': 16, 'FEED_SIZE_LIMIT': 32},
            {'name': 'postproc', 'stage_processing_handler': postproc_stage, 'BATCH_SIZE': 16, 'FEED_SIZE_LIMIT': 32},
            {'name': 'save', 'stage_processing_handler': save_stage, 'BATCH_SIZE': 16, 'FEED_SIZE_LIMIT': 32},
        ]
    }

    pipeline = Pipeline()
    pipeline.build_pipeline(pipeline_config)

    input_handler = FilenameInputHandler()
    pipeline.set_input_handler(input_handler)
    output_handler = NoOpOutputHandler()
    pipeline.set_output_handler(output_handler)
    
    # if not os.path.exists(dst_basedir):
    #     os.makedirs(dst_basedir)
    input_handler.set({})
    load_stage.set({'base_path': image_dir, 'num_cpu': 4})
    preproc_stage.set({})
    inference_stage.set({})
    postproc_stage.set({})
    save_stage.set({'base_path': output_dir, 'num_cpu': 4})
    output_handler.set({})
    
    filename_list = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpg')]
    pipeline.feed_sync(filename_list) 
