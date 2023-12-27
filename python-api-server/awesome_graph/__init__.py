import json
import logging
import colorlog

def get_configure(project_name):
    with open(f'./datarun/{project_name}/config.json', 'r') as file:
        global_config = json.load(file)
    global_config['project_name'] = project_name
    return global_config


def get_logger():
    logger = colorlog.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a handler for writing the log into a file
    fh = logging.FileHandler('end_to_end.log')
    fh.setLevel(logging.DEBUG)

    # Specify the colour scheme for log levels
    handler = logging.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)-8s%(reset)s %(message)s', datefmt=None, reset=True, log_colors={'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'red,bg_white', }, style='%'))

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(handler)
    return logger
logger = get_logger()