
import os
import logging

from logzero import setup_logger


def get_logger(log_name=None,
               logfile_name=None,
               output_dir='exp_default',
               level=logging.INFO):
    """获取logger
    """
    if not logfile_name:
        logfile_name = 'default.log'
    logger_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    logger = setup_logger(
        name=log_name,
        logfile=os.path.join(logger_dir, logfile_name),
        level=level,
        maxBytes=4096 * 1000,  # 4MB by default and use rotate file handler
        backupCount=10)
    return logger
