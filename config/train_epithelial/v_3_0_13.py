class GeneratorConfig:
    final_sigmoid = True
    f_maps = 64
    layer_order = 'gcr'
    num_groups = 8
    num_levels = 5
    is_segmentation = True
    conv_padding = 1
    pool_kernel_size = [(1, 2, 2)]

class DiscriminatorConfig:
    n_layers_start = 1
    n_layers_end = 2
    data_per_new_layer = 80_000
    fade_in_ratio = 1

class Config:
    batch_size = 8
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
    datasetB_file = 'epithelial_v_1_2_0.h5'
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
        {'name': 'RandomGaussianNoise', 'mean': 0, 'std': 0.2, 'mask_condition': '1, (x[0] != -1)'},
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
    gan_mode = 'lsgan'
    generator_config = GeneratorConfig()
    generator_output_range = (0, 1)
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
    lr = 0.0002
    lr_decay_iters = 50
    lr_policy = 'linear'
    max_dataset_size = float('inf')
    model = 'cycle_gan_3d'
    n_epochs = 50
    n_epochs_decay = 50
    n_layers_D = 3
    name = 'epithelial_sheets_v_3_0_13'
    ndf = 16
    netD = 'progressive'
    netG = 'UNet3D'
    ngf = 64
    no_dropout = True
    no_flip = False
    no_html = False
    no_normalization = False
    norm = 'batch'
    num_threads = 1
    output_nc = 2
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
