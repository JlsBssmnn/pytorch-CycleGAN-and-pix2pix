output_nc = 1

class CycleGANGeneratorConfig:
    unet_1x2x2_kernel_scale = True
    unet_extra_xy_conv = False
    last_activation = 'tanh'

class UNet3DGeneratorConfig:
    final_sigmoid = True
    f_maps = 64
    layer_order = 'gcr'
    num_groups = 8
    num_levels = 5
    is_segmentation = True
    conv_padding = 1
    pool_kernel_size = [(1, 2, 2)]

class DiscriminatorConfig:
    disc_1x2x2_kernel_scale = False
    disc_extra_xy_conv = False
    disc_no_decrease_last_layers = False
    last_activation = None

class EvaluationConfig:
    class GeneratorConfig:
        output_nc = output_nc

    slice_str = '0, 16'
    membrane_black = False
    ground_truth_file = './datasets/epithelial_sheet/evaluation/ground_truth.h5'
    ground_truth_datasets = [('slice1_membrane_truth', 'slice1_cell_truth'),
                             ('slice2_membrane_truth', 'slice2_cell_truth'),
                             ('slice3_membrane_truth', 'slice3_cell_truth')]
    image_names = ['slice1', 'slice2', 'slice3']
    input_file = './datasets/epithelial_sheet/trainA/real_image.h5'
    input_dataset = 'image'
    image_slices = [':, 23:55, 1034:1546, 1031:1543', ':, 31:63, 1941:2453, 54:566', ':, 52:84, 35:547, 2029:2541']
    patch_size = (32, 64, 64)
    generator_config = GeneratorConfig()
    use_gpu = True
    
    stride = (16, 32, 32)
    batch_size = 256
    basins_range = (10, 11, 1)
    membrane_range = (225, 226, 1)
    error_factor = 1/2
    show_progress = False
    local_error_measure = 'acc'
    local_error_a = 0.5
    local_error_b = 3.5

class Config:
    batch_size = 64
    beta1 = 0.5
    border_offset_A = [0, 5, 18]
    border_offset_B = [0, 0, 0]
    checkpoints_dir = './checkpoints'
    continue_train = False
    crop_size = 256
    dataroot = './datasets/epithelial_sheet'
    datasetA_file = 'real_image.h5'
    datasetA_mask = 'mask'
    datasetA_names = None
    datasetB_file = 'epithelial_v_1_3_0.h5'
    datasetB_mask = None
    datasetB_names = None
    dataset_length = 'min'
    dataset_mode = 'unaligned_3d'
    dataset_stride = None
    dataset_transforms_A = [
        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
        {'name': 'RandomFlip'},
    ]
    dataset_transforms_B = [
        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
        {'name': 'RandomFlip'},
        {'name': 'ChannelModifier', 'x': 0, 'channels_to_delete': [1]},
    ]
    direction = 'AtoB'
    disc_transforms_A = None
    disc_transforms_B = None
    discriminator_config = DiscriminatorConfig()
    display_env = 'main'
    display_freq = 9_984
    display_id = -1
    display_ncols = 4
    display_port = 8097
    display_server = 'http://localhost'
    display_winsize = 256
    element_size_um = [0.5, 0.2, 0.2]
    epoch = 'latest'
    epoch_count = 1
    evaluation_config = EvaluationConfig()
    eval_freq = 1_024
    eval_params = {'compute_VI': False}
    gan_mode = 'lsgan'
    generator_config = CycleGANGeneratorConfig()
    generator_output_range = (-1, 1)
    gpu_ids = [0]
    init_gain = 0.02
    init_type = 'normal'
    input_nc = 1
    input_value_range = (0, 255)
    isTrain = True
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_identity = 0
    load_iter = 0
    load_size = 286
    lr_D = 0.0002
    lr_G = 0.0002
    lr_decay_iters = 50
    lr_policy = 'linear'
    max_dataset_size = float('inf')
    model = 'cycle_gan_3d'
    n_epochs = 35
    n_epochs_decay = 35
    n_layers_D = 3
    name = 'epithelial_sheets_grid_search_0'
    ndf = 64
    netD = 'basic'
    netG = 'unet_32x64x64'
    ngf = 64
    no_dropout = True
    no_flip = False
    no_html = False
    no_normalization = False
    norm = 'batch'
    num_threads = 1
    output_nc = output_nc
    phase = 'train'
    pool_size = 50
    post_transforms_A = [
        {'name' : 'Threshold', 'lower': -1, 'upper': 1, 'slice_string': '0:1'},
    ]
    post_transforms_B = None
    preprocess = 'resize_and_crop'
    print_freq = 128
    real_fake_trans_A = None
    real_fake_trans_B = None
    sample_size = [32, 64, 64]
    save_by_iter = False
    save_epoch_freq = 99999
    save_latest_freq = 2_496
    serial_batches = False
    suffix = ''
    update_html_freq = display_freq
    use_wandb = False
    verbose = False
    wandb_project_name = 'CycleGAN-and-pix2pix'

config = Config()

# 72 combinations
grid_params = [
    {
        # 3 combinations
        'parameter': 'generator_config',
        'complex_values': [
            {
                'value': CycleGANGeneratorConfig(),
                'other_values': {
                    'batch_size': [8, 64],
                    'generator_output_range': (-1, 1),
                    'evaluation_config.batch_size': 128,
                    'netG': 'unet_32x64x64',
                }
            },
            {
                'value': UNet3DGeneratorConfig(),
                'other_values': {
                    'batch_size': 8,
                    'generator_output_range': (0, 1),
                    'evaluation_config.batch_size': 8,
                    'netG': 'UNet3D',
                }
            },
        ],
    },
    {
        # 6 combinations
        'parameter': 'datasetB_file',
        'complex_values': [
            {
                'value': 'epithelial_v_1_1_1.h5',
                'other_values': {
                    'evaluation_config.membrane_black': True,
                    'dataset_transforms_B': [[
                        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
                        {'name': 'RandomFlip'},
                    ]],
                    'output_nc': 1,
                    'evaluation_config.generator_config.output_nc': 1,
                    'disc_transforms_A': [[
                            {'name': 'RandomPixelModifier', 'change_probability': 0.025, 'value': -1},
                            {'name': 'RandomPixelModifier', 'change_probability': 0.15, 'value': 1},
                        ],
                        None
                    ],
                    'evaluation_config.basins_range': (100, 256, 25),
                    'evaluation_config.membrane_range': (0, 150, 5),
                }
            },
            {
                'value': 'epithelial_v_1_2_0.h5',
                'other_values': {
                    'evaluation_config.membrane_black': True,
                    'dataset_transforms_B': [[
                        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
                        {'name': 'RandomFlip'},
                        {'name': 'RandomGaussianNoise', 'mean': 0, 'std': 0.2, 'mask_condition': '1, (x[0] != -1)'},
                    ]],
                    'output_nc': 2,
                    'evaluation_config.generator_config.output_nc': 2,
                    'disc_transforms_A': [[
                            {'name': 'RandomPixelModifier', 'change_probability': 0.025, 'value': -1},
                            {'name': 'RandomPixelModifier', 'change_probability': 0.15, 'value': 1},
                        ],
                        None
                    ],
                    'evaluation_config.basins_range': (100, 256, 25),
                    'evaluation_config.membrane_range': (0, 150, 5),
                }
            },
            {
                'value': 'epithelial_v_1_3_0.h5',
                'other_values': {
                    'evaluation_config.membrane_black': False,
                    'dataset_transforms_B': [[
                        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
                        {'name': 'RandomFlip'},
                        {'name': 'ChannelModifier', 'x': 0, 'channels_to_delete': [1]},
                    ]],
                    'output_nc': 1,
                    'evaluation_config.generator_config.output_nc': 1,
                    'disc_transforms_A': [[
                            {'name': 'RandomPixelModifier', 'change_probability': 0.15, 'value': -1},
                            {'name': 'RandomPixelModifier', 'change_probability': 0.025, 'value': 1},
                        ],
                        None
                    ],
                    'evaluation_config.basins_range': (0, 150, 5),
                    'evaluation_config.membrane_range': (100, 256, 25),
                }
            },
        ]
    },
    {
        # 2 combinations
        'parameter': 'lr_D',
        'values': [0.0002, 0.00001],
    },
    {
        # 2 combinations
        'parameter': 'lambda_A',
        'complex_values': [
            {
                'value': 5.0,
                'other_values': {
                    'lambda_B': 5.0,
                }
            },
            {
                'value': 10.0,
                'other_values': {
                    'lambda_B': 10.0,
                }
            },
        ],
    },
]
