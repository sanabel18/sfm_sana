import logging
import sys
from os.path import join
from datetime import datetime
from pytz import timezone, utc


def get_logger(name, save_path, tz="Asia/Taipei", debug=False):
    ''' Modified from videoio v2.0.1 written by qhan.'''
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter("[%(levelname)-.1s %(asctime)s %(name)-21s] %(message)s", \
                    datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # output logs to file
    file_handler = logging.FileHandler(join(save_path, name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def custom_time(*arg):
        utc_dt = utc.localize(datetime.utcnow())
        converted = utc_dt.astimezone(timezone(tz))
        return converted.timetuple()
        
    logging.Formatter.converter = custom_time

    return logger


def close_all_handlers(logger: logging.Logger):
    handlers = logger.handlers[:]  # Reason: You should not iterate over a list and remove elements from it, you'll end up skipping elements.
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)
    print(f'{logger.name}: All handlers are closed and removed.')