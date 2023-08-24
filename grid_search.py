import argparse
from functools import reduce
import importlib
from itertools import product
import os
import train
import json
from copy import deepcopy
from pathlib import Path
from util.logging_config import logging
import bdb

from util.my_utils import object_to_dict

class Encoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def getattr_r_no_except(obj, attr):
    """
    Recursively get an attribute of an object. Different attributes can be separated by a period.
    This function will not throw an attribute error if the attribute doesn't exist. Instead, it
    return the attribute error.
    """
    parts = attr.split('.')
    assert len(parts) >= 1

    if not hasattr(obj, parts[0]):
        return AttributeError(f"'{obj.__class__.__name__}' object has no attribute '{parts[0]}'")
    attribute = getattr(obj, parts[0])

    for part in parts[1:]:
        if not hasattr(attribute, part):
            return AttributeError(f"'{attribute.__class__.__name__}' object has no attribute '{part}'")
        attribute = getattr(attribute, part)
    return attribute

def getattr_r(obj, attr):
    """
    Recursively get an attribute of an object. Different attributes can be separated by a period.
    """
    parts = attr.split('.')
    assert len(parts) >= 1
    attribute = getattr(obj, parts[0])

    for part in parts[1:]:
        attribute = getattr(attribute, part)
    return attribute

def setattr_r(obj, attr, value):
    """
    Recursively set an attribute of an object. Different attributes can be separated by a period.
    """
    parts = attr.split('.')
    assert len(parts) >= 1
    attribute = obj

    for part in parts[:-1]:
        attribute = getattr(attribute, part)

    setattr(attribute, parts[-1], value)

class GridParameterManager:
    def __init__(self, config, grid):
        self.config = config
        self.grid = grid
        self.combinations = []
        self.create_param_combinations()

        self.project_dir = Path(config.checkpoints_dir) / config.name
        os.makedirs(self.project_dir, exist_ok=True)

        self.config_indices = list(range(len(self.combinations)))
        digit_dirs = [int(x) for x in os.listdir(self.project_dir) if x.isdigit()]
        self.config_indices = [x for x in self.config_indices if x not in digit_dirs]

    def create_param_combinations(self):
        for param in self.grid:
            param_combinations = []
            if 'values' in param:
                for value in param['values']:
                    param_combinations.append({param['parameter']: value})
            elif 'complex_values' in param:
                for complex_value in param['complex_values']:
                    keys = [param['parameter']]
                    values = [[complex_value['value']]]

                    for key, value in complex_value['other_values'].items():
                        keys.append(key)
                        if type(value) == list:
                            values.append(value)
                        else:
                            values.append([value])
                    for param_combination in product(*values):
                        param_combinations.append({k: v for k, v in zip(keys, param_combination)})
            else:
                raise ValueError('Each parameter must have either `values` or `complex_values`!')

            self.combinations.append(param_combinations)
        
        self.combinations = [reduce(lambda a, b: a | b, x) for x in product(*self.combinations)]

    def create_config(self, config, params):
        for param_name, param_value in params.items():
            setattr_r(config, param_name, param_value)
        return config

    def config_iter(self):
        for i in self.config_indices:
            params = self.combinations[i]
            experiment_dir = self.project_dir / str(i)
            os.makedirs(experiment_dir, exist_ok=True)

            config = self.create_config(deepcopy(self.config), params)
            config.name = str(i)
            config.checkpoints_dir = self.project_dir
            with open(experiment_dir / 'config.json', 'x') as f:
                json.dump(object_to_dict(config), f, indent=4, sort_keys=True, cls=Encoder)

            logging.info(f'starting experiment {i}')

            yield config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='The config file')
    parser.add_argument('--test', action='store_true', help='Creates a test run which only does one iteration per grid combination')
    parser.add_argument('--run_iterations', default=None, type=int, help='How many iterations shall be run, default is every combination')

    args = parser.parse_args()

    config_module = importlib.import_module(f"config.{args.config}")
    base_config = config_module.config
    grid_params = config_module.grid_params

    grid_manager = GridParameterManager(base_config, grid_params)

    i = 0
    for config in grid_manager.config_iter():
        try:
            train.main(config, args.test)
        except Exception as e:
            if isinstance(e, bdb.BdbQuit):
                exit()
            logging.error('Error occurred during experiment:')
            logging.error(e)
        finally:
            i += 1
            if args.run_iterations is not None and i >= args.run_iterations:
                break

if __name__ == '__main__':
    main()
