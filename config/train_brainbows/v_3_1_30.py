from data.dynamic_dataset_creation import create_offsets, synth_brainbow_to_affinities
from util.visualizing_transforms import AffinityToSeg, ToUint8

offsets = create_offsets(8, 18)
input_nc = 4
output_nc = len(offsets) + 1
sample_size = [32, 64, 64]

class GeneratorConfig:
    unet_1x2x2_kernel_scale = True
    unet_extra_xy_conv = False
    last_activation = 'tanh'

class DiscriminatorConfig:
    disc_1x2x2_kernel_scale = True
    disc_extra_xy_conv = False
    disc_no_decrease_last_layers = True
    last_activation = None

class EvaluationConfig:
    class GeneratorConfig:
        output_nc = output_nc

    batch_size = 128
    bg_measure = 'mean'
    bg_threshold = 0.5
    bg_vi_weight = 0.5
    bias_cut_range = (-0.3, 0.11, 0.1)
    dist_measure = 'norm'
    eval_freq = 200
    generator_config = GeneratorConfig()
    ground_truth_dataset = 'ground_truth'
    ground_truth_file = './datasets/brainbows/ground_truth.h5'
    ground_truth_slices = ['0:32, 384:512, 384:512', '0:32, 0:128, 0:128', '68:100, 384:512, 0:128']
    image_names = ['slice1', 'slice2', 'slice3']
    image_slices = [':, 0:32, 384:512, 384:512', ':, 0:32, 0:128, 0:128', ':, 68:100, 384:512, 0:128']
    image_type = 'affinity'
    input_dataset = 'color_image'
    input_file = './datasets/brainbows/trainA/real_image.h5'
    offsets = offsets
    patch_size = (32, 64, 64)
    scale_with_patch_max = True
    seperating_channel = 3
    show_progress = False
    slice_str = ':'
    use_gpu = True
    vi_freq = 5000

class Config:
    batch_size = 5
    beta1 = 0.5
    border_offset_A = [0, 0, 0]
    border_offset_B = [0, 0, 0]
    checkpoints_dir = './checkpoints'
    continue_train = False
    crop_size = 256
    dataroot = './datasets/brainbows'
    datasetA_creation_func = None
    datasetA_file = 'real_image.h5'
    datasetA_mask = 'mask'
    datasetA_names = ['color_image']
    datasetA_random_sampling = True
    datasetB_creation_func = lambda _, image: synth_brainbow_to_affinities(image, offsets)
    datasetB_file = 'brainbows_v_1_3_1.h5'
    datasetB_mask = None
    datasetB_names = ['color']
    datasetB_random_sampling = True
    dataset_length = 'min'
    dataset_mode = 'unaligned_3d'
    dataset_stride = (16, 32, 32)
    dataset_transforms_A = [
        {'name': 'Scaler', 'in_min': -1, 'in_max': 1, 'out_min': 0, 'out_max': 1},
        {'name': 'RandomDiscreteRotation', 'angles': [0, 90, 180, 270]},
        {'name': 'RandomFlip'},
        {'name': 'RandomCropAndZoom', 'crop_ratio': 1.5, 'sample_size': sample_size},
        {'name': 'RandomColorJitter', 'brightness': [0.5, 1], 'contrast': 0.5, 'saturation': 0.5, 'hue': 0.25},
        {'name': 'RandomOpening'},
        {'name': 'RandomClosing'},
        {'name': 'RandomGaussianBlur', 'channels': input_nc, 'kernel_size': [1, 3, 3], 'sigma_range': (0, 1, 0.1)},
        {'name': 'RandomSharpening', 'sharpness_factor': 2},
        {'name': 'Scaler', 'in_min': 0, 'in_max': 1, 'out_min': -1, 'out_max': 1},
    ]
    dataset_transforms_B = None
    direction = 'AtoB'
    disc_transforms_A = [
        {'name': 'GaussianBlur', 'channels': output_nc, 'kernel_size': [1, 3, 3], 'sigma': 0.3},
    ]
    disc_transforms_B = None
    discriminator_config = DiscriminatorConfig()
    display_env = 'main'
    display_freq = 5_000
    display_id = -1
    display_ncols = 4
    display_port = 8097
    display_server = 'http://localhost'
    display_transform_A = ToUint8
    display_transform_B = AffinityToSeg
    display_winsize = 256
    element_size_um = [0.2, 0.1, 0.1]
    epoch = 'latest'
    epoch_count = 1
    evaluation_config = EvaluationConfig()
    extra_losses = [
        {'name': 'AffinityConsistencyLoss', 'factor': 5},
    ]
    gan_mode = 'lsgan'
    generator_config = GeneratorConfig()
    generator_output_range = (-1, 1)
    gpu_ids = [0]
    init_gain = 0.02
    init_type = 'normal'
    input_nc = input_nc
    input_value_range = (0, 255)
    isTrain = True
    lambda_A = 5.0
    lambda_B = 5.0
    lambda_identity = 0
    load_iter = 0
    load_size = 286
    lr_D = 0.0002
    lr_G = 0.0002
    lr_decay_iters = 50
    lr_policy = 'linear'
    max_dataset_size = 2_000
    model = 'cycle_gan_3d'
    n_epochs = 50
    n_epochs_decay = 50
    n_layers_D = 5
    name = 'brainbows_v_3_1_30'
    ndf = 64
    netD = 'n_layers'
    netG = 'unet_32x64x64'
    ngf = 64
    no_adversarial_loss = False
    no_dropout = True
    no_flip = False
    no_html = False
    no_normalization = False
    norm = 'batch'
    num_threads = 1
    output_nc = output_nc
    phase = 'train'
    pool_size = 50
    post_transforms_A = disc_transforms_A
    post_transforms_B = None
    preprocess = 'resize_and_crop'
    print_freq = 100
    real_fake_trans_A = None
    real_fake_trans_B = None
    sample_size = sample_size
    save_by_iter = False
    save_epoch_freq = 50
    save_latest_freq = 5000
    serial_batches = False
    suffix = ''
    update_html_freq = display_freq
    use_tensorboard = True
    use_wandb = False
    verbose = False
    wandb_project_name = 'CycleGAN-and-pix2pix'

config = Config()
