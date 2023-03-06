import unittest

import torch
from models.cycle_gan_3d_model import CycleGAN3dModel

from test.test_utils import tmp_dir

class TestOptions:
    beta1 = 0.5
    checkpoints_dir = tmp_dir
    dataset_mode = 'unaligned_3d'
    direction = 'AToB'
    gan_mode = 'lsgan'
    gpu_ids = [0]
    init_gain = 0.02
    init_type = 'normal'
    input_nc = 1
    isTrain = True
    lambda_A = 10.0
    lambda_B = 10.0
    lambda_identity = 0.5
    lr = 0.0002
    n_layers_D = 3
    name = 'test'
    ndf = 64
    netD = 'basic'
    netG = 'resnet_9blocks'
    ngf = 64
    no_dropout = None
    norm = 'instance'
    output_nc = 1
    pool_size = 50
    preprocess = 'resize_and_crop'

class CycleGAN3dModelTest(unittest.TestCase):

    def test_training_step(self):
        model = CycleGAN3dModel(TestOptions())
        shape = (1, 40, 40, 40)
        model.set_input({
            'A': torch.arange(torch.tensor(shape).prod(), dtype=torch.float).reshape(shape),
            'B': torch.arange(torch.tensor(shape).prod(), dtype=torch.float).reshape(shape),
            'A_image': 0,
            'B_image': 0,
        })
        model.optimize_parameters()
        self.assertEqual(model.real_A.shape, shape)
        self.assertEqual(model.real_B.shape, shape)
        self.assertEqual(model.fake_A.shape, shape)
        self.assertEqual(model.fake_B.shape, shape)
        self.assertEqual(model.rec_A.shape, shape)
        self.assertEqual(model.rec_B.shape, shape)
        self.assertEqual(model.idt_A.shape, shape)
        self.assertEqual(model.idt_B.shape, shape)
