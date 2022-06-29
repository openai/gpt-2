import logging

def make_logger(name: str):
    logging.basicConfig()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    return logger