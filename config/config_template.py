from typing import Any, Callable

Transform = list[dict[str, Any]]

class CUGeneratorConfig:
    unet_1x2x2_kernel_scale: bool # Use a kernel that has half the size in z-direction compared to x and y
    unet_extra_xy_conv: bool      # If true, add an extra layer that just scales down in x- and y-direction
    last_activation: str | None   # The last activation function of the generator

class U3DGeneratorConfig:
    final_sigmoid: bool # whether to end the network with a sigmoid layer
    f_maps: int # needed for computing the number of channels in the network, higher number -> more channels
    layer_order: str # how norm, activation and convolution are organized
    num_groups: int # parameter for group norm
    num_levels: int # how deep the network will be
    is_segmentation: bool # if true, the network architecture will be adjusted approprietly
    conv_padding: int # padding for convolutional layers
    pool_kernel_size: list[tuple[int, int, int]] # a list of kernel sizes for each layer, not every layer must be specified

class DiscriminatorConfig:
    disc_1x2x2_kernel_scale: bool      # Use a kernel that has half the size in z-direction compared to x and y
    disc_extra_xy_conv: bool           # If true, add an extra layer that just scales down in x- and y-direction
    disc_no_decrease_last_layers: bool # If true, the last layer doesn't scale down its input
    last_activation: str | None        # The last activation function of the discriminator

class Config:
    batch_size: int # Training batch size
    beta1: float # beta 1 for adam optimizer
    border_offset_A: list[int] # Specify how many voxels in z-, y- and x- direction shall be omitted for dataset A for training
    border_offset_B: list[int] # Specify how many voxels in z-, y- and x- direction shall be omitted for dataset B for training
    checkpoints_dir: str # directory where results of the experiment will be stored
    continue_train: bool # If true, attempted to continue training
    dataroot: str # directory where the two datasets are located
    datasetA_creation_func: None | Callable # An optional function that can modify dataset A
    datasetA_file: str # name of the file used as dataset A
    datasetA_mask: str | None # an optional h5 dataset name that contains a mask; the mask will be applied to dataset A
    datasetA_names: list[str] | None # names of h5 datasets that contain the training data; if None, all datasets are used
    datasetA_random_sampling: bool # if true, samples are randomly sampled from the image during training
    datasetB_creation_func: None | Callable # An optional function that can modify dataset B
    datasetB_file: str # name of the file used as dataset B
    datasetB_mask: str | None # an optional h5 dataset name that contains a mask; the mask will be applied to dataset B
    datasetB_names: list[str] | None # names of h5 datasets that contain the training data; if None, all datasets are used
    datasetB_random_sampling: bool # if true, samples are randomly sampled from the image during training
    dataset_length: str # how len of dataset is determined, supports 'min' and 'max'
    dataset_mode: str # which dataset implementation is used
    dataset_stride: None | list[int] # the dataset stride, if none the sample size is used (only relevant when not randomly sampling)
    dataset_transforms_A: None | Transform # if provided, samples are passed through the transform immediately after being extracted
    dataset_transforms_B: None | Transform # if provided, samples are passed through the transform immediately after being extracted
    direction: str # whether to go from A to B or B to A
    disc_transforms_A: None | Transform # if provided, real samples are passed through the transform before being fed into the discriminator
    disc_transforms_B: None | Transform # if provided, real samples are passed through the transform before being fed into the discriminator
    discriminator_config: DiscriminatorConfig
    display_freq: int # at how many iterations the visualizer is active
    display_id: int # property for the visdom server
    display_ncols: int # property for the visdom server
    display_port: int # property for the visdom server
    display_server: str # property for the visdom server
    display_transform_A: Callable # a function from visualizing_transforms.py which converts generator output to an image format
    display_transform_B: Callable # a function from visualizing_transforms.py which converts generator output to an image format
    display_winsize: int # property for the visdom server
    element_size_um: list[float] # the real size of a voxel across z-, y- and x-direction
    epoch_count: int # which epoch training will start
    evaluation_config: None | Any # config specifying how to evaluate during training
    extra_losses: Transform # extra losses that are used
    gan_mode: str # which GAN loss is used
    generator_config: CUGeneratorConfig | U3DGeneratorConfig
    generator_output_range: tuple[float, float] # between which values the generator output lies
    gpu_ids: list[int] # ids of GPUs that are used
    init_gain: float # scaling factor for network initialization
    init_type: str # type of network initialization
    input_nc: int # input number of channels
    input_value_range: tuple[float, float] # between which values the input image values lie
    isTrain: bool # whether to train or test
    lambda_A: float # the lambda for the cycle A -> B -> A
    lambda_B: float # the lambda for the cycle B -> A -> B
    lambda_identity: float # weight for identity loss
    lr_D: float # learning rate for discriminators
    lr_G: float # learning rate for generators
    lr_decay_iters: int # how many epochs to decay the learning rate
    lr_policy: str # how to decay the learning rate
    max_dataset_size: int # maximum number of samples for one epoch
    model: str # the model used for training
    n_epochs: int # number of epochs without decay
    n_epochs_decay: int # number of epochs with decay
    n_layers_D: int # number layers for discriminator
    name: str # name of experiment
    ndf: int # number of discriminator features, determines complexity of network
    netD: str # the discriminator architecture
    netG: str # the generator architecture
    ngf: int # number of generator features, determines complexity of network
    no_adversarial_loss_A: bool # whether to disable the adversarial loss for generator A
    no_adversarial_loss_B: bool # whether to disable the adversarial loss for generator B
    no_dropout: bool # if false, dropout is used for the generator
    no_html: bool # if true, no images are saved
    no_normalization: bool # if true, samples are not normalized
    norm: str # which norm to use for generator and discriminator
    num_threads: int # number of threads for data loading
    output_nc: int # number of output channels
    partial_cycle_A: bool # if true, only generator B is optimized from the cycle A -> B -> A
    partial_cycle_B: bool # if true, only generator A is optimized from the cycle B -> A -> B
    phase: str # which phase, e.g. train, eval
    pool_size: int # pool size for fakes that are sampled for the discriminator
    post_transforms_A: None | Transform # if provided, generated samples are passed through the transform before being fed into the discriminator 
    post_transforms_B: None | Transform # if provided, generated samples are passed through the transform before being fed into the discriminator 
    preprocess: str | list[str] # preprocessing for original CycleGAN
    print_freq: int # after how many iterations to print metrics
    real_fake_trans_A: None | Transform # if provided, real samples are converted using the function and fed into the discriminator as fakes
    real_fake_trans_B: None | Transform # if provided, real samples are converted using the function and fed into the discriminator as fakes
    sample_size: list[int] # the size of a sample in z-, y- and x-direction
    save_by_iter: bool # if true, models are stored for an iteraion instead of just the latest
    save_epoch_freq: int # after how many epochs to save the models
    save_latest_freq: int # after how many epochs to save the models as latest
    scale_with_patch_max: bool # if true, samples are scaled via the max in the sample instead of the max in the entire image
    serial_batches: bool # if true, samples from B are sapmled sequentially
    update_html_freq: int # at how many iterations images are saved
    use_tensorboard: bool # if true, tensorboard is used
    use_transformed_cycle: bool # if true, images transformed by the post_transform are fed into the other generator for the cycle
    use_transformed_disc: bool # if true, images transformed by the post_transform are fed into the discriminator
    use_wandb: bool # if true, weights and biases will be used
    verbose: bool # if true, more debug information is printed
    wandb_project_name: str # name of wandb project

config = Config()
