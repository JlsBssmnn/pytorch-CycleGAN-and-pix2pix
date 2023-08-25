import argparse
from collections import defaultdict
from copy import deepcopy
import importlib
from itertools import chain
from pathlib import Path
import pandas as pd

root = Path(__file__).parent.parent

from grid_search import GridParameterManager
from helpers.config_fix import get_config_param, params_cyclegan_unet, params_3d_unet, params_disc, params_dataset, params_loss, params_network, params_transforms, params_training

parameters = [
    'name',
    *params_cyclegan_unet,
    *params_3d_unet,
    *params_disc,
    *params_dataset,
    *params_loss,
    *params_network,
    *params_transforms,
    *params_training,
]

name_mapping = {
        'generator_config.unet_1x2x2_kernel_scale': 'gen_1x2x2_kernel', 'generator_config.unet_extra_xy_conv': 'gen_extra_xy_conv',
        'generator_config.last_activation': 'gen_last_activation', 'generator_config.final_sigmoid': 'gen_final_sigmoid',
        'generator_config.f_maps': 'gen_fmaps', 'generator_config.layer_order': 'gen_layer_order',
        'generator_config.num_groups': 'gen_num_groups', 'generator_config.num_levels': 'gen_num_levels',
        'generator_config.is_segmentation': 'gen_is_segmentation', 'generator_config.conv_padding': 'gen_conv_padding',
        'generator_config.pool_kernel_size': 'gen_pool_kernel', 'discriminator_config.disc_1x2x2_kernel_scale':
        'disc_1x2x2_kernel', 'discriminator_config.disc_extra_xy_conv': 'disc_extra_xy_conv',
        'discriminator_config.disc_no_decrease_last_layers': 'disc_no_decrease_ll',
        'discriminator_config.last_activation': 'disc_last_activation'
}

def get_target_parameter_names(parameters):
    return [name_mapping[param] if param in name_mapping else param for param in parameters]

def get_parameters(category):
    match category:
        case 'training':
            return get_target_parameter_names(params_training)
        case 'transforms':
            return get_target_parameter_names(params_transforms)
        case 'network':
            return get_target_parameter_names(params_network)
        case 'loss':
            return get_target_parameter_names(params_loss)
        case 'dataset':
            return get_target_parameter_names(params_dataset)
        case 'disc':
            return get_target_parameter_names(params_disc)
        case '3d_unet':
            return get_target_parameter_names(params_3d_unet)
        case 'cyclegan_unet':
            return get_target_parameter_names(params_cyclegan_unet)

def convert_to_tuple(config, config_path, grid_search_index=None, module=None):
    name = 'e_' if config_path.parent.stem == 'train_epithelial' else 'b_'
    name += 'gs_' + str(grid_search_index) if config_path.stem.startswith('grid_search') else config_path.stem

    config_item = [get_config_param(config, param, module, config_path) for param in parameters]
    config_item[0] = name

    return config_item

def main():
    parser = argparse.ArgumentParser(description='Summarizes all config files into a cdf file')
    parser.add_argument('output_file', type=str, help='The file where the information is written to')
    args = parser.parse_args()

    data = defaultdict(lambda: [])
    def insert_config(config):
        for i, param in enumerate(parameters):
            param = name_mapping[param] if param in name_mapping else param
            data[param].append(config[i])

    for config_path in chain((root / 'config/train_epithelial').iterdir(), (root / 'config/train_brainbows').iterdir()):
        if config_path.stem == '__pycache__':
            continue

        config_dir = config_path.parent.parent.parent
        module_name = str(config_path.relative_to(config_dir)).replace('\\', '.').replace('/', '.').replace('.py', '')

        if config_path.stem.startswith('grid_search'):
            config_module = importlib.import_module(module_name)
            base_config = config_module.config
            grid_params = config_module.grid_params
            manager = GridParameterManager(base_config, grid_params)

            for i in manager.config_indices:
                params = manager.combinations[i]
                config = manager.create_config(deepcopy(manager.config), params)
                config = convert_to_tuple(config, config_path, i)
                insert_config(config)
        else:
            module = importlib.import_module(module_name)
            config = module.config
            config = convert_to_tuple(config, config_path, module=module)
            insert_config(config)
    
    df = pd.DataFrame(data)
    df.to_csv(args.output_file)
