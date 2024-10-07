import math
import numpy as np
import cv2
import functools
import tensorflow.compat.v1 as tf
try:
    import tensorflow.contrib.tensorrt as trt
    # Make things with supposed existing module
except:
    pass

class PersonMaskModel: # Segmentation model in fact
    def __init__(self, config):
        self.config = config
        self.person_label = self.config['person_label']
        self.BATCH_SIZE = 1
        self.MAX_INPUT_SIZE = tuple(map(int, self.config['MAX_INPUT_SIZE']))
        model_path = self.config['MODEL_PATH']
        with tf.gfile.GFile(model_path, "rb") as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        try:
            self.graph_def = trt.create_inference_graph(input_graph_def=self.graph_def, \
                                                        outputs=['ImageTensor:0', 'SemanticPredictions:0'], \
                                                        max_batch_size=self.BATCH_SIZE, \
                                                        max_workspace_size_bytes=2<<32, \
                                                        precision_mode='FP16')
        except:
            pass

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(self.graph_def, name='')
        self.inputs = self.graph.get_tensor_by_name('ImageTensor:0')
        self.outputs = self.graph.get_tensor_by_name('SemanticPredictions:0')
        self.zero_input = np.zeros((self.MAX_INPUT_SIZE[1], self.MAX_INPUT_SIZE[0], 3), dtype=np.uint8)

    def start(self): 
        if 'USE_CPU' in self.config and self.config['USE_CPU']:
            intra_op_parallelism_threads = 1
            if 'INTRA_OP_THREAD' in self.config:
                intra_op_parallelism_threads = int(self.config['INTRA_OP_THREAD'])
            inter_op_parallelism_threads = 1
            if 'INTER_OP_THREAD' in self.config:
                inter_op_parallelism_threads = int(self.config['INTER_OP_THREAD'])
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
                intra_op_parallelism_threads=intra_op_parallelism_threads,
                inter_op_parallelism_threads=inter_op_parallelism_threads
            ))
        else:
            gpu_options = tf.GPUOptions(allow_growth=True)
            self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            # first feed to occupy gpu
            batch_inputs = [self.zero_input] * self.BATCH_SIZE
            self.inference(batch_inputs)

    def preproc(self, frame_list):
        pad_frame_list = []
        valid_hw_list = []
        for frame in frame_list:
            h, w, _ = frame.shape
            resize_ratio = max(math.ceil(h / float(self.MAX_INPUT_SIZE[1])), math.ceil(w / float(self.MAX_INPUT_SIZE[0])))
            if resize_ratio > 1.0:
                frame = cv2.resize(frame, (0,0), fx=1.0/resize_ratio, fy=1.0/resize_ratio) 
            pad_frame = np.zeros((self.MAX_INPUT_SIZE[1], self.MAX_INPUT_SIZE[0], 3), dtype=np.uint8)
            h, w, _ = frame.shape
            pad_frame[0:h, 0:w, :] = frame
            pad_frame_list.append(pad_frame)
            valid_hw_list.append((h,w))


        return pad_frame_list, valid_hw_list
    
    def inference(self, pad_frame_list):
        batch_outputs = []
        i = 0
        end = len(pad_frame_list) - self.BATCH_SIZE
        while i < len(pad_frame_list):
            feed_dict = {self.inputs: pad_frame_list[i: i+self.BATCH_SIZE]}
            _batch_outputs = self.sess.run(self.outputs, feed_dict=feed_dict)
            i += self.BATCH_SIZE
            batch_outputs.extend(_batch_outputs)

        return batch_outputs

    def postproc(self, frame_list, valid_hw_list, mask_list):
        result_mask_list = []
        for frame, valid_hw, mask in zip(frame_list, valid_hw_list, mask_list):
            mask = mask[:valid_hw[0], :valid_hw[1]]
            mask = mask == self.person_label
            area = np.sum(mask)
            mask = (mask * 255).astype(np.uint8) 
            pad = int(np.sqrt(area)) // 10
            if pad > 1:
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad,pad)))
            mask = 255 - mask
            # resize mask to original frame size
            h, w, _ = frame.shape
            mask_rs = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)

            result_mask_list.append(mask_rs)

        return result_mask_list
        
