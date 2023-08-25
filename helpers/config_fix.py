'''
Fixes a config by checking which attributes are not up-to-date or missing and
adjusts them accordingly. Thus, older configs can also be used for training.
'''

import re
import inspect
from typeguard import TypeCheckError, check_type
import importlib
from config.config_template import Config as ConfigTemplate
from util.my_utils import getattr_r_no_except, setattr_r
from util.visualizing_transforms import ToUint8

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
        case 'border_offset_A' | 'border_offset_B' :
            return [0, 0, 0]
        case 'datasetA_random_sampling' | 'datasetB_random_sampling':
            return False
        case 'datasetA_creation_func' | 'datasetB_creation_func' | 'dataset_stride' | 'dataset_transforms_A' | 'dataset_transforms_B':
            return None
        case 'disc_transforms_A' | 'disc_transforms_B':
            return None
        case 'display_transform_A' | 'display_transform_B':
            return ToUint8
        case 'element_size_um':
            return [0.5, 0.2, 0.2] if 'epithelial' in config_path else [0.2, 0.1, 0.1]
        case 'extra_losses':
            return []
        case 'evaluation_config':
            return None
        case 'generator_output_range':
            return (-1, 1)
        case 'input_value_range':
            return (0, 255)
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
        case 'post_transforms_A' | 'real_fake_trans_A' | 'post_transforms_B' | 'real_fake_trans_B':
            return None
        case 'scale_with_patch_max':
            return False if 'epithelial' in config_path else True
        case 'use_tensorboard':
            return False
        case 'use_transformed_cycle' | 'use_transformed_disc':
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

def fix_config(config_module: str):
    config = importlib.import_module(config_module).config

    for attr, attr_type in inspect.get_annotations(ConfigTemplate).items():
        if inspect.isclass(attr_type) and attr_type.__module__ != 'builtins':
            for sub_attr, sub_attr_type in inspect.get_annotations(attr_type).items():
                try:
                    check_type(getattr(getattr(config, attr), sub_attr), sub_attr_type)
                except (TypeCheckError, AttributeError):
                    missing = get_config_param(config, attr + '.' + sub_attr, None, config_module)
                    setattr_r(config, attr + '.' + sub_attr, missing)
        else:
            try:
                check_type(getattr(config, attr), attr_type)
            except (TypeCheckError, AttributeError):
                missing = get_config_param(config, attr, None, config_module)
                setattr(config, attr, missing)
    # def verify_with_base_class(obj):
    #     assert len(obj.__class__.__bases__) == 1
    #     base_class = obj.__class__.__bases__[0]

    #     for attr, attr_type in inspect.get_annotations(base_class).items():
    #         check_type(getattr(obj, attr), attr_type)

    # verify_with_base_class(config)
    # config_attrs = set([x for x in vars(config.__class__).keys() if not x.startswith('__')])
    # for attr in config_attrs:
    #     verify_with_base_class(getattr(config, attr))

    return config
