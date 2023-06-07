output_nc = 1

class GeneratorConfig:
    unet_1x2x2_kernel_scale = True
    unet_extra_xy_conv = False
    last_activation = 'tanh'

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
    image_slices = [':, 30:62, 1034:1546, 1031:1543', ':, 38:70, 1941:2453, 54:566', ':, 59:91, 35:547, 2029:2541']
    patch_size = (32, 64, 64)
    generator_config = GeneratorConfig()
    use_gpu = True
    
    stride = (16, 32, 32)
    batch_size = 128
    basins_range = (0, 150, 5)
    membrane_range = (100, 256, 25)
    error_factor = 1/2
    show_progress = False
    local_error_measure = 'acc'
    local_error_a = 0.5
    local_error_b = 3.5

class Config:
    batch_size = 128
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
    eval_freq = 2_496
    gan_mode = 'lsgan'
    generator_config = GeneratorConfig()
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
    n_epochs = 50
    n_epochs_decay = 50
    n_layers_D = 3
    name = 'epithelial_sheets_v_4_0_11'
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
    save_epoch_freq = 5
    save_latest_freq = 5000
    serial_batches = False
    suffix = ''
    update_html_freq = display_freq
    use_wandb = False
    verbose = False
    wandb_project_name = 'CycleGAN-and-pix2pix'

config = Config()
