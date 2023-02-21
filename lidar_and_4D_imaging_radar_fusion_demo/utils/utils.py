# -- coding: utf-8 --
import os
from loguru import logger


def set_log_dir(log_path):
    if not os.path.exists(log_path):
        logger.info("created log path!")
        os.makedirs(log_path)
    filename = os.path.join(log_path, 'train_log.txt')
    logger.add(filename)