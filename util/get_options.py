import argparse
from helpers.config_fix import fix_config
from options.train_options import TrainOptions
from util.logging_config import logging
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def get_training_options():
    """
    Decides whether to read in a config file or use the command-line arguments
    as options. This function returns the read options.
    """
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.add_argument('config', help='The config file which shall be used')
    parser.add_argument('--test', action='store_true', help='Do just a test run')

    try:
        with suppress_stdout_stderr():
            args = parser.parse_args()
        config = fix_config('config.' + args.config)
        logging.info('Loaded configuration %s', args.config)
        return config, args.test
    except SystemExit:
        return TrainOptions().parse(), False   # get training options
