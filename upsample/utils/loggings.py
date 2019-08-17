'''
various stuffs for logging
'''
from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO, WARN
formatter = Formatter('%(asctime)s %(name)s[%(levelname)s] %(message)s', "%Y-%m-%d %H:%M:%S")

def get_standard_logger(name):
    logger = getLogger(name)
    logger.setLevel(INFO)
    handler = StreamHandler()
    handler.setLevel(logger.getEffectiveLevel())
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
