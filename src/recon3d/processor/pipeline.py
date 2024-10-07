from abc import ABC, abstractmethod
import os, sys
import time
import threading
from collections import deque
import traceback


# process one iteratable at one time
class InputHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set(self, config):
        pass
    
    @abstractmethod
    def process(self, input):
        # process one iteratable at one time
        # return output dict !!
        pass


class FilenameInputHandler(InputHandler):
    def __init__(self):
        pass
    
    def set(self, config):
        pass

    def process(self, input):
        filename = input.rstrip()

        return {'filename': filename} 


class NoOpInputHandler(InputHandler):
    def __init__(self):
        pass
    
    def set(self, config):
        pass

    def process(self, input):
        return input


class PipelineEntry:
    def __init__(self, FEED_SIZE_LIMIT):
        self.name = 'Input'
        self.FEED_SIZE_LIMIT = FEED_SIZE_LIMIT

        self.input_handler = None

        self.next_stage_trigger = threading.Event()
        self.done_trigger = threading.Event()
        self.output_buffer = deque()
        
        self.next_stage_consume_trigger = None 

    def reset(self):
        self.next_stage_trigger.clear()
        self.done_trigger.clear()
        self.output_buffer.clear()

    def set_next_stage(self, next_stage):
        self.next_stage_consume_trigger = next_stage.consume_trigger

    def set_input_handler(self, input_handler):
        self.input_handler = input_handler

    def feed_nonsync(self, input_list):
        for _input in input_list:
            while len(self.output_buffer) > self.FEED_SIZE_LIMIT:
                self.next_stage_consume_trigger.wait() 
                self.next_stage_consume_trigger.clear() 
            self.output_buffer.append(self.input_handler.process(_input))
            self.next_stage_trigger.set()

    def trigger_wait(self):
        # trick !! the order
        self.done_trigger.set()
        self.next_stage_trigger.set()

    def feed_sync(self, input_list):
        self.feed_nonsync(input_list)
        self.trigger_wait()

    def monitor_buffer(self):
        return '[ {:10s}: {:5d} ]'.format(self.name, len(self.output_buffer))


class StageProcessingHandler(ABC):
    @abstractmethod
    def __init__(self, inferencer_resource):
        pass

    @abstractmethod
    def set(self, stage_config):
        pass
    
    # @abstractmethod
    # def get_batch_size(self): # for default value
    #     pass

    @abstractmethod
    def process(self, batch):
        # return list of output dict
        pass


class PipelineStage:
    def __init__(self, name, prev_pipeline_stage, stage_processing_handler, BATCH_SIZE, FEED_SIZE_LIMIT):
        self.name = name
        # from previous stage
        self.trigger = prev_pipeline_stage.next_stage_trigger
        self.prev_stage_done_trigger = prev_pipeline_stage.done_trigger
        self.input_buffer = prev_pipeline_stage.output_buffer
        self.stage_processing_handler = stage_processing_handler 
        self.BATCH_SIZE = BATCH_SIZE
        if self.BATCH_SIZE is None:
            self.BATCH_SIZE = self.stage_processing_handler.get_batch_size() # for default value
            assert self.BATCH_SIZE is not None
        self.FEED_SIZE_LIMIT = FEED_SIZE_LIMIT
        if self.FEED_SIZE_LIMIT is None:
            self.FEED_SIZE_LIMIT = 3 * self.BATCH_SIZE
        

        # share with PipelineEntry
        self.next_stage_trigger = threading.Event()
        self.done_trigger = threading.Event()
        self.output_buffer = deque()

        self.consume_trigger = threading.Event()
        self.next_stage_consume_trigger = None
        prev_pipeline_stage.set_next_stage(self)

        self.running_flag = False

    def start(self):
        if self.running_flag:
            return
        self.running_flag = True
        self.thread = threading.Thread(name=self.name, target=self.stage_process)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        if not self.running_flag:
            return
        self.running_flag = False
        self.trigger.set()
        self.thread.join()
    
    def reset(self):
        self.next_stage_trigger.clear()
        self.done_trigger.clear()
        self.output_buffer.clear()
        self.consume_trigger.clear()

    def set_next_stage(self, next_stage):
        self.next_stage_consume_trigger = next_stage.consume_trigger
    
    def stage_process(self):
        try:
            while self.running_flag:
                self.trigger.wait()
                self.trigger.clear()
                
                while self.input_buffer: # not empty
                    batch = []
                    vacancy = self.BATCH_SIZE
                    while self.input_buffer and vacancy > 0:
                        batch.append(self.input_buffer.popleft())
                        vacancy -= 1
                    while len(self.output_buffer) > self.FEED_SIZE_LIMIT:
                        self.next_stage_consume_trigger.wait() 
                        self.next_stage_consume_trigger.clear()
                    output_list = self.stage_processing_handler.process(batch)
                    self.consume_trigger.set()
                    self.output_buffer.extend(output_list)
                    self.next_stage_trigger.set()

                if self.prev_stage_done_trigger.isSet() and not self.input_buffer:
                    # trick !! the order
                    self.done_trigger.set()
                    self.next_stage_trigger.set()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(str(e))
            os._exit(1)

    def monitor_buffer(self):
        return '[ {:10s}: {:5d} ]'.format(self.name, len(self.output_buffer))

    def get_name(self):
        return self.name


class OutputHandler(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def set(self, config):
        pass

    @abstractmethod
    def _extract(self, info):
        # return info to keep
        pass
    
    @abstractmethod
    def process(self, output):
        # process one output at one time
        # return None
        pass


class NoOpOutputHandler(OutputHandler):
    def __init__(self):
        pass

    def set(self, config):
        pass

    def _extract(self, info):
        pass
    
    def process(self, output):
        pass


class PipelineOutput:
    def __init__(self, prev_pipeline_stage):
        # from previous stage
        self.trigger = prev_pipeline_stage.next_stage_trigger
        self.prev_stage_done_trigger = prev_pipeline_stage.done_trigger
        self.input_buffer = prev_pipeline_stage.output_buffer
        self.output_handler = None
        self.done_trigger = threading.Event()
        self.consume_trigger = threading.Event()
        prev_pipeline_stage.set_next_stage(self)
        self.running_flag = False

    def set_output_handler(self, output_handler):
        self.output_handler = output_handler
    
    def start(self):
        if self.running_flag:
            return
        self.running_flag = True
        self.thread = threading.Thread(name='output', target=self.stage_process)
        self.thread.setDaemon(True)
        self.thread.start()

    def stop(self):
        if not self.running_flag:
            return
        self.running_flag = False
        self.trigger.set()
        self.thread.join()

    def reset(self):
        self.done_trigger.clear()

    def stage_process(self):
        try:
            while self.running_flag:
                self.trigger.wait()
                self.trigger.clear()
                while self.input_buffer: # not empty
                    self.output_handler.process(self.input_buffer.popleft())
                    self.consume_trigger.set()
                if self.prev_stage_done_trigger.isSet() and not self.input_buffer:
                    self.done_trigger.set()
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(str(e))
            os._exit(1)

    def wait(self):
        self.done_trigger.wait()
    
    def is_done(self):
        return self.done_trigger.isSet()


class Pipeline:
    def __init__(self):
        self.input_stage = None
        self.ordered_stage = None
        self.stage_dict = None
        self.output_stage = None
        self.running_flag = False
        self.MONITOR_BUFFER = False
        self.running_monitor_flag = False
        self.monitor_buffer_period = -1

    def __del__(self):
        # what to do
        pass
        
    def build_pipeline(self, pipeline_config):
        FEED_SIZE_LIMIT = 2048
        if 'FEED_SIZE_LIMIT' in pipeline_config:
            FEED_SIZE_LIMIT = pipeline_config['FEED_SIZE_LIMIT']
        stage_config = pipeline_config['stage_config']
        # stage_config: [{'name': 'LoadImgnp', ...}, {'name': 'DetectFace', ...}, ...]
        # can add stage

        self.input_stage = PipelineEntry(FEED_SIZE_LIMIT)
        self.ordered_stage = []
        self.stage_dict = dict()
        prev_stage = self.input_stage
        for stage in stage_config:
            cur_stage = PipelineStage(stage['name'], prev_stage, stage['stage_processing_handler'], stage['BATCH_SIZE'], stage['FEED_SIZE_LIMIT'])

            self.ordered_stage.append(cur_stage)
            self.stage_dict[stage['name']] = cur_stage
            prev_stage = cur_stage
        self.output_stage = PipelineOutput(prev_stage)

        if 'MONITOR_BUFFER_PERIOD' in os.environ:
            self.MONITOR_BUFFER = True
            self.running_monitor_flag = False
            self.monitor_buffer_period = int(os.environ['MONITOR_BUFFER_PERIOD'])

            print('============')
            print('stage config')
            for stage in self.ordered_stage:
                print('    ||')
                print('    \/')
                print('[ '+stage.get_name()+' ]')

            
    def set_stage_processing_handler(self, name, stage_config):
        self.stage_dict[name].stage_processing_handler.set(stage_config)
    
    
    def start(self):
        if self.running_flag:
            return
        self.running_flag = True
        for stage in self.ordered_stage:
            stage.start()
        self.output_stage.start()
        if self.MONITOR_BUFFER:
            self.running_monitor_flag = True
            self.monitor_buffer_thread = threading.Thread(name="monitor_buffer_thread", target=self.run_monitor_buffer)
            self.monitor_buffer_thread.setDaemon(True)
            self.monitor_buffer_thread.start()

    def stop(self):
        if not self.running_flag:
            return
        self.running_flag = False
        for stage in self.ordered_stage:
            stage.stop()
        self.output_stage.stop()
        if self.MONITOR_BUFFER:
            self.running_monitor_flag = False
            self.monitor_buffer_thread.join()

    def reset(self):
        self.input_stage.reset()
        for stage in self.ordered_stage:
            stage.reset()
        self.output_stage.reset()

    def get_ordered_stage(self):
        return self.ordered_stage
        
    def set_input_handler(self, input_handler):
        self.input_stage.set_input_handler(input_handler)

    def set_output_handler(self, output_handler):
        self.output_stage.set_output_handler(output_handler)

    def wait_sync(self):
        self.output_stage.wait()
        self.reset()
    
    def is_done(self):
        return self.output_stage.is_done()

    def run_monitor_buffer(self):
        try:
            while not self.is_done() and self.running_monitor_flag:
                monitor_str = ' | '.join([self.input_stage.monitor_buffer()] + [stage.monitor_buffer() for stage in self.ordered_stage])
                print(monitor_str)
                time.sleep(self.monitor_buffer_period)
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            print(str(e))
            os._exit(1)

    def feed_sync(self, input_list):
        self.start()
        self.input_stage.feed_sync(input_list)
        self.wait_sync()
        self.stop()

    def feed_sync_efficient(self, input_list):
        self.input_stage.feed_sync(input_list)
        self.wait_sync()

    ######################
    # start()
    # feed_nonsync()
    # feed_nonsync()
    # ....
    # wait_nonsync()
    # ....
    # feed_nonsync()
    # feed_nonsync()
    # ....
    # wait_nonsync()
    # stop()
    def feed_nonsync(self, input_list):
        self.input_stage.feed_nonsync(input_list)

    def wait_nonsync(self):
        self.input_stage.trigger_wait()
        self.output_stage.wait()
        self.reset()