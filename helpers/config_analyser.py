import argparse
from collections import defaultdict
from copy import deepcopy
import importlib
from itertools import chain
from pathlib import Path
import re
import pandas as pd

root = Path(__file__).parent.parent

from grid_search import GridParameterManager, getattr_r_no_except

params_cyclegan_unet = ['generator_config.unet_1x2x2_kernel_scale', 'generator_config.unet_extra_xy_conv', 'generator_config.last_activation']
params_3d_unet = ['generator_config.final_sigmoid', 'generator_config.f_maps', 'generator_config.layer_order', 'generator_config.num_groups',
                  'generator_config.num_levels', 'generator_config.is_segmentation', 'generator_config.conv_padding', 'generator_config.pool_kernel_size']
params_disc = ['discriminator_config.disc_1x2x2_kernel_scale', 'discriminator_config.disc_extra_xy_conv',
               'discriminator_config.disc_no_decrease_last_layers', 'discriminator_config.last_activation']
params_dataset = ['datasetA_random_sampling', 'datasetB_random_sampling', 'datasetB_creation_func', 'datasetB_file',
                  'dataset_stride', 'max_dataset_size', 'offsets', 'sample_size', 'serial_batches']
params_loss = [
    'extra_losses', 'gan_mode',
    'lambda_A', 'lambda_B', 'lambda_identity',
    'partial_cycle_A', 'partial_cycle_B',
    'no_adversarial_loss_A', 'no_adversarial_loss_B',
]
params_network = ['ndf', 'ngf', 'netD', 'netG', 'n_layers_D', 'no_dropout', 'norm']
params_transforms = ['disc_transforms_A', 'disc_transforms_B', 'post_transforms_A', 'real_fake_trans_A', 'dataset_transforms_A', 'dataset_transforms_B']
params_training = ['batch_size', 'lr_D', 'lr_G', 'n_epochs', 'n_epochs_decay', 'use_transformed_cycle']

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
    
def grep_line(file_name, pattern):
    with open(file_name) as f:
        matching_lines = [line for line in f if re.search(pattern, line)]
        assert len(matching_lines) == 1
        line = matching_lines[0]
    return line

def get_missing_param(config, param, module, config_path):
    match param:
        case 'offsets':
            if module is not None and hasattr(module, 'offsets'):
                return grep_line(config_path, '^offsets = create_offsets')[len('offsets = '):].replace('\n', '')
            else:
                return None
        case _ if param in params_cyclegan_unet:
            if config.netG == 'UNet3D' or config.netG == 'ResidualUNet3D' or config.netG == 'ResidualUNetSE3D':
                return None
            else:
                match param:
                    case 'generator_config.unet_1x2x2_kernel_scale':
                        if hasattr(config, 'unet_1x2x2_kernel_scale'):
                            return getattr(config, 'unet_1x2x2_kernel_scale')
                        return False
                    case 'generator_config.unet_extra_xy_conv':
                        if hasattr(config, 'unet_extra_xy_conv'):
                            return getattr(config, 'unet_extra_xy_conv')
                        return False
                    case 'generator_config.last_activation':
                        return 'tanh'
                    case _:
                        raise NotImplementedError(f"Handling of parameter {param} not handled")
        case _ if param in params_3d_unet:
            if config.netG.startswith('unet') or config.netG.startswith('resnet'):
                return None
            else:
                match param:
                    case 'generator_config.final_sigmoid':
                        return True
                    case 'generator_config.f_maps':
                        return 64
                    case 'generator_config.layer_order':
                        return 'generator_config.gcr'
                    case 'generator_config.num_groups':
                        return 8
                    case 'generator_config.num_levels':
                        return 4
                    case 'generator_config.is_segmentation':
                        return True
                    case 'generator_config.conv_padding':
                        return 1
                    case 'generator_config.pool_kernel_size':
                        return 2
                    case _:
                        raise NotImplementedError(f"Handling of parameter {param} not handled")
        case _ if param in params_disc:
            match param:
                case 'discriminator_config.disc_1x2x2_kernel_scale' | 'discriminator_config.disc_extra_xy_conv' | 'discriminator_config.disc_no_decrease_last_layers':
                    return False
                case 'discriminator_config.last_activation':
                    return None
                case _:
                    raise NotImplementedError(f"Handling of parameter {param} not handled")
        case 'datasetA_random_sampling' | 'datasetB_random_sampling':
            return False
        case 'datasetB_creation_func' | 'dataset_stride' | 'dataset_transforms_A' | 'dataset_transforms_B':
            return None
        case 'disc_transforms_A' | 'disc_transforms_B':
            return None
        case 'extra_losses':
            return []
        case 'lr_D' | 'lr_G':
            return config.lr
        case 'no_adversarial_loss_A' | 'partial_cycle_A':
            post_transforms_A = get_config_param(config, 'post_transforms_A', module, config_path)
            if post_transforms_A is None:
                return False
            else:
                return any([transform == 'threshold' or transform['name'] == 'Threshold' for transform in post_transforms_A])
        case 'no_adversarial_loss_B' | 'partial_cycle_B':
            return False
        case 'post_transforms_A' | 'real_fake_trans_A':
            return None
        case 'use_transformed_cycle':
            return len(get_config_param(config, 'post_transforms_A', module, config_path) or []) > 0
        case _:
            raise NotImplementedError(f"Handling of parameter {param} not handled")

def get_config_param(config, param, module, config_path):
    attr = getattr_r_no_except(config, param)
    if isinstance(attr, AttributeError):
        return get_missing_param(config, param, module, config_path)
    
    match param:
        case 'datasetB_creation_func' if attr is not None:
            line = grep_line(config_path, 'datasetB_creation_func = lambda')
            line = line[line.find(':') + 2:].replace('\n', '')
            return line

    return attr

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
