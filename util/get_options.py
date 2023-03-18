import sys
import importlib
from options.train_options import TrainOptions
from util.logging_config import logging

def get_training_options():
    """
    Decides whether to read in a config file or use the command-line arguments
    as options. This function returns the read options.
    """
    if len(sys.argv) < 2:
        print('Missing arguments, see --help or use `config <path-to-config>` to use a config file')
        exit(1)
    if sys.argv[1] == 'config':
        if len(sys.argv) < 3:
            print('The path to the config file is missing')
            exit(1)
        logging.info('Loaded configuration %s', sys.argv[2])
        return importlib.import_module('config.' + sys.argv[2]).config
    else:
        return TrainOptions().parse()   # get training options
