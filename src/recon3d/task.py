import subprocess
import time
import traceback
import os

from utils.logger import get_logger, close_all_handlers

class TaskResult(object):
    def __init__(self, returncode: int, time: float, logfile: str):
        self.returncode = returncode
        self.time = time
        self.logfile = logfile

class Task(object):
    def __init__(self, step: int, log_dir: str):
        ''' 
        Initialization for all kinds of task:
            - Setup the logger
        '''

        self.log_name = '_'.join([f'step{step:02d}', type(self).__name__])
        self.logger = get_logger(self.log_name + '.log', log_dir)
        self.logfile = os.path.join(log_dir, self.log_name + '.log') 

    def run_task_content(self):
        raise NotImplementedError(f'run_task_content not implemented')
        
    def run(self) -> TaskResult:
        ''' Wrapper of the actual content to be run, which is different from task to task. '''
        
        t = time.time()
        self.logger.info('Task started.')
        try:
            returncode = self.run_task_content()
            if returncode == 0:
                self.logger.info('Task ended successfully.')
            else:
                self.logger.info('Task failed.')
        except KeyboardInterrupt:
            self.logger.warning('Process cancelled by user.')
            returncode = -1
        # except Exception as e:
        except Exception:
            e = traceback.format_exc()
            self.logger.error(f'Task failed: {e}')
            returncode = -1
        finally:
            t = time.time() - t
            self.logger.info(f'Task duration: {t}')
            close_all_handlers(self.logger)
            return TaskResult(returncode, t, self.logfile)
