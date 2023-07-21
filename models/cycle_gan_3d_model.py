"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>: Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import itertools
from models import custom_transforms
from models.custom_transforms import create_transform
from util.image_pool import ImagePool
from util.my_utils import object_to_dict
from util.Evaluater import BrainbowEvaluater, EpithelialEvaluater
from .base_model import BaseModel
from . import networks_3d


class CycleGAN3dModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='epithelial')
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--unet_1x2x2_kernel_scale', action='store_true', default=False, help='If using certain unets, this parameter is passed to the skip blocks of the unet (for setting kernel size)')
        parser.add_argument('--unet_extra_xy_conv', action='store_true', default=False, help='If using certain unets, this parameter is passed to the skip blocks of the unet (for adding an extra conv layer)')
        if is_train:
            parser.add_argument('eval_freq', type=int, default=100, help='How often the evaluation shall be computed')
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--post_transforms_A', type=dict, default=None, help='Transformations that are applied to the output of Gen_A to create fake_B')
            parser.add_argument('--post_transforms_B', type=dict, default=None, help='Transformations that are applied to the output of Gen_B to create fake_A')
            parser.add_argument('--disc_transforms_A', type=dict, default=None, help='Transformations that are applied to the real image before it is given to discriminator A')
            parser.add_argument('--disc_transforms_B', type=dict, default=None, help='Transformations that are applied to the real image before it is given to discriminator B')
            parser.add_argument('--real_fake_trans_A', type=dict, default=None, help='Transformations that are applied to the real image, but the result represents a fake image (for discriminator A)')
            parser.add_argument('--real_fake_trans_B', type=dict, default=None, help='Transformations that are applied to the real image, but the result represents a fake image (for discriminator B)')
            parser.add_argument('--partial_cycle_A', type=bool, default=False, help='If true, the cycle loss A (A -> B -> A) will only be used to optimize generator B')
            parser.add_argument('--partial_cycle_B', type=bool, default=False, help='If true, the cycle loss B (B -> A -> B) will only be used to optimize generator A')
            parser.add_argument('--use_transformed_cycle', type=bool, default=True, help='If true, the result from the post transform is fed into the other generator to compute the reconstruction, otherwise the direct output from the generator is used')

            parser.add_argument('--disc_1x2x2_kernel_scale', action='store_true', default=False, help='Passed to NLayerDiscriminator')
            parser.add_argument('--disc_extra_xy_conv', action='store_true', default=False, help='Passed to NLayerDiscriminator')
            parser.add_argument('--disc_no_decrease_last_layers', action='store_true', default=False, help='Passed to NLayerDiscriminator')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        if opt.evaluation_config is None:
            self.evaluater = None
        elif 'epithelial_sheet' in opt.dataroot:
            self.evaluater = EpithelialEvaluater(opt)
        elif 'brainbows' in opt.dataroot:
            self.evaluater = BrainbowEvaluater(opt)
        else:
            raise NotImplementedError(f"For a dataroot {opt.dataroot}, there is no evaluater!")

        assert opt.dataset_mode == 'unaligned_3d'
        self.loss_names = ['cycle_A', 'cycle_B']
        if not opt.no_adversarial_loss_A:
            self.loss_names += ['G_A', 'D_A']
        if not opt.no_adversarial_loss_B:
            self.loss_names += ['G_B', 'D_B']

        self.use_aff_con = False
        self.cycle_weight_threshold = None
        for extra_loss in opt.extra_losses:
            if extra_loss['name'] == 'AffinityConsistencyLoss':
                self.loss_names += ['aff_con']
                self.use_aff_con = True
                self.criterionAffCon = networks_3d.AffinityConsistencyLoss(opt)
                self.aff_con_factor = extra_loss['factor']
            elif extra_loss['name'] == 'weigted_cycle_loss':
                self.cycle_weight_threshold = extra_loss['threshold']

        # specify the images you want to save and display.
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if opt.post_transforms_A is not None:
            visual_names_A += ['raw_fake_B']
        if opt.post_transforms_B is not None:
            visual_names_B += ['raw_fake_A']

        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
            self.loss_names += ['idt_A', 'idt_B']

        if opt.datasetA_creation_func is not None:
            visual_names_A.append('original_A')
        if opt.datasetB_creation_func is not None:
            visual_names_B.append('original_B')
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks_3d.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        **object_to_dict(opt.generator_config))
        self.netG_B = networks_3d.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        **object_to_dict(opt.generator_config))

        self.generator_output_f = custom_transforms.Scaler(opt.generator_output_range[0], opt.generator_output_range[1], -1, 1)

        if self.isTrain:  # define discriminators
            self.netD_A = networks_3d.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            **object_to_dict(opt.discriminator_config))
            self.netD_B = networks_3d.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids,
                                            **object_to_dict(opt.discriminator_config))
            self.post_transform_A = create_transform(opt.post_transforms_A)
            self.post_transform_B = create_transform(opt.post_transforms_B)

            self.disc_transform_A = create_transform(opt.disc_transforms_A)
            self.disc_transform_B = create_transform(opt.disc_transforms_B)

            self.real_fake_trans_A = create_transform(opt.real_fake_trans_A) if opt.real_fake_trans_A else None
            self.real_fake_trans_B = create_transform(opt.real_fake_trans_B) if opt.real_fake_trans_B else None

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks_3d.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.

            self.criterionCycle = torch.nn.L1Loss()
            self.criterionCycleNoReduction = torch.nn.L1Loss(reduction='none')

            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def get_current_losses(self, total_iters):
        if self.evaluater is None:
            return super().get_current_losses(total_iters)
        return super().get_current_losses(total_iters) | self.evaluater.compute_evaluation(self.netG_A, total_iters)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_index = input['A_image' if AtoB else 'B_image']

        if 'original_A' in input:
            self.original_A = input['original_A'].detach().clone()
        if 'original_B' in input:
            self.original_B = input['original_B'].detach().clone()

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.generator_output_f(self.netG_A(self.real_A))
        if self.opt.partial_cycle_A:
            self.raw_fake_B = self.fake_B.detach().clone()
            self.fake_B = self.post_transform_A(self.fake_B).detach()
        else:
            self.raw_fake_B = self.fake_B.clone()
            self.fake_B = self.post_transform_A(self.fake_B)
        self.rec_A = self.generator_output_f(self.netG_B(self.fake_B if self.opt.use_transformed_cycle else self.raw_fake_B))   # G_B(G_A(A))

        self.fake_A = self.generator_output_f(self.netG_B(self.real_B))
        if self.opt.partial_cycle_B:
            self.raw_fake_A = self.fake_A.detach().clone()
            self.fake_A = self.post_transform_B(self.fake_A).detach()
        else:
            self.raw_fake_A = self.fake_A.clone()
            self.fake_A = self.post_transform_B(self.fake_A)
        self.rec_B = self.generator_output_f(self.netG_A(self.fake_A if self.opt.use_transformed_cycle else self.raw_fake_A))   # G_A(G_B(B))

    def batch_processed(self, batch_size):
        if self.opt.netD == 'progressive':
            self.netD_A.module.batch_processed(batch_size)
            self.netD_B.module.batch_processed(batch_size)

    def backward_D_basic(self, netD, real, fake, synthetic_fake=None):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # synthetic fake
        if synthetic_fake is not None:
            pred_synthetic_fake = netD(synthetic_fake.detach())
            loss_D_synthetic_fake = self.criterionGAN(pred_synthetic_fake, False)

        # Combined loss and calculate gradients
        if synthetic_fake is None:
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D = (loss_D_real + loss_D_fake + loss_D_synthetic_fake) * (1/3)

        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, _ = networks_3d.cal_gradient_penalty(netD, real, fake, netD.src_device_obj, lambda_gp=self.opt.lambda_gp)
            loss_D += gradient_penalty
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        if self.real_fake_trans_A:
            synthetic_fake_B = self.real_fake_trans_A(self.real_B)
        else:
            synthetic_fake_B = None
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.disc_transform_A(self.real_B.clone()), fake_B, synthetic_fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        if self.real_fake_trans_B:
            synthetic_fake_A = self.real_fake_trans_B(self.real_A)
        else:
            synthetic_fake_A = None
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.disc_transform_B(self.real_A.clone()), fake_A, synthetic_fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.generator_output_f(self.netG_A(self.real_B))
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.generator_output_f(self.netG_B(self.real_A))
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.opt.no_adversarial_loss_A:
            self.loss_G_A = 0
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)

        if self.opt.no_adversarial_loss_B:
            self.loss_G_B = 0
        else:
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        if self.cycle_weight_threshold is not None:
            self.loss_cycle_B = self.criterionCycleNoReduction(self.rec_B, self.real_B) * lambda_B
            weights = networks_3d.compute_cycle_weight(self.real_B, self.cycle_weight_threshold)
            self.loss_cycle_B = (self.loss_cycle_B * weights).mean()
        else:
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients

        if self.use_aff_con:
            self.loss_aff_con = self.criterionAffCon(self.fake_B) * self.aff_con_factor
        else:
            self.loss_aff_con = 0

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + \
                      self.loss_idt_B + self.loss_aff_con
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        if not self.opt.no_adversarial_loss_A:
            self.backward_D_A()      # calculate gradients for D_A
        if not self.opt.no_adversarial_loss_B:
            self.backward_D_B()      # calculate gradients for D_B

        self.optimizer_D.step()  # update D_A and D_B's weights
