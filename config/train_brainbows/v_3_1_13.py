class GeneratorConfig:
    unet_1x2x2_kernel_scale = True
    unet_extra_xy_conv = False
    last_activation = 'tanh'

class DiscriminatorConfig:
    n_layers_start = 3
    n_layers_end = 5
    data_per_new_layer = 50_000
    fade_in_ratio = 0.5

class Config:
    batch_size = 5
    beta1 = 0.5
    border_offset_A = [0, 0, 0]
    border_offset_B = [0, 0, 0]
    checkpoints_dir = './checkpoints'
    continue_train = False
    crop_size = 256
    dataroot = './datasets/brainbows'
    datasetA_file = 'real_image.h5'
    datasetA_mask = 'mask'
    datasetA_names = ['color_image']
    datasetB_file = 'brainbows_v_1_3_1.h5'
    datasetB_mask = None
    datasetB_names = ['color']
    dataset_length = 'min'
    dataset_mode = 'unaligned_3d'
    dataset_stride = (16, 32, 32)
    dataset_transforms_A = [
        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
        {'name': 'RandomFlip'},
    ]
    dataset_transforms_B = [
        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
        {'name': 'RandomFlip'},
    ]
    direction = 'AtoB'
    disc_transforms_A = None
    disc_transforms_B = None
    discriminator_config = DiscriminatorConfig()
    display_env = 'main'
    display_freq = 10_000
    display_id = -1
    display_ncols = 4
    display_port = 8097
    display_server = 'http://localhost'
    display_winsize = 256
    element_size_um = [0.2, 0.1, 0.1]
    epoch = 'latest'
    epoch_count = 1
    gan_mode = 'lsgan'
    generator_config = GeneratorConfig()
    generator_output_range = (-1, 1)
    gpu_ids = [0]
    init_gain = 0.02
    init_type = 'normal'
    input_nc = 4
    input_value_range = (0, 255)
    isTrain = True
    lambda_A = 5.0
    lambda_B = 5.0
    lambda_identity = 5
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
    n_layers_D = 5
    name = 'brainbows_v_3_1_13'
    ndf = 64
    netD = 'progressive'
    netG = 'unet_32x64x64'
    ngf = 64
    no_dropout = True
    no_flip = False
    no_html = False
    no_normalization = False
    norm = 'batch'
    num_threads = 1
    output_nc = 4
    phase = 'train'
    pool_size = 50
    post_transforms_A = None
    post_transforms_B = None
    preprocess = 'resize_and_crop'
    print_freq = 100
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
