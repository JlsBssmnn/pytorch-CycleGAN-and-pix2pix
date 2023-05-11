import functools
import unittest
import torch
import torch.nn as nn

from models.networks_3d import ResnetGenerator, UnetGenerator, NLayerDiscriminator, ProgressiveDiscriminator
from models.custom_transforms import Scaler

class Resnet9Blocks3dTest(unittest.TestCase):

  def test_output_dimensions(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    net = ResnetGenerator(1, 1, 64, norm_layer=norm_layer, use_dropout=True, n_blocks=9)

    expected_shape = (1, 32, 32, 32)
    image = torch.arange(torch.tensor(expected_shape).prod(), dtype=torch.float).reshape(expected_shape)
    out = net(image)
    self.assertEqual(out.shape, expected_shape)

    expected_shape = (1, 40, 40, 40)
    image = torch.arange(torch.tensor(expected_shape).prod(), dtype=torch.float).reshape(expected_shape)
    out = net(image)
    self.assertEqual(out.shape, expected_shape)

    expected_shape = (1, 100, 100, 100)
    image = torch.arange(torch.tensor(expected_shape).prod(), dtype=torch.float).reshape(expected_shape)
    out = net(image)
    self.assertEqual(out.shape, expected_shape)
    
    expected_shape = (1, 16, 16, 16)
    image = torch.arange(torch.tensor(expected_shape).prod(), dtype=torch.float).reshape(expected_shape)
    out = net(image)
    self.assertEqual(out.shape, expected_shape)


class UnetGenerator3dTest(unittest.TestCase):
  def test_correct_shape(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    net = UnetGenerator(1, 1, 5, norm_layer=norm_layer, use_dropout=True)

    net_input = torch.rand((1, 1, 32, 64, 64))
    net_output = net(net_input)
    self.assertEqual(net_output.shape, (1, 1, 32, 64, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 1, 32, 32, 32))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 1, 32, 32, 32))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 1, 64, 128, 64))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 1, 64, 128, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 1, 32, 64, 64))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 1, 32, 64, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 1, 32, 32, 32))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 1, 32, 32, 32))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((5, 1, 64, 128, 64))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (5, 1, 64, 128, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

  def test_UnetSkipConnectionBlock_1x2x2(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    net = UnetGenerator(1, 1, 6, norm_layer=norm_layer, use_dropout=True, unet_1x2x2_kernel_scale=True, unet_extra_xy_conv=True)

    net_input = torch.rand((1, 1, 32, 64, 64))
    net_output = net(net_input)
    self.assertEqual(net_output.shape, (1, 1, 32, 64, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    inner_shape = None
    next_shape = None
    def save_inner_shape(_, __, output):
      nonlocal inner_shape
      inner_shape = output.shape
    def save_next_shape(_, __, output):
      nonlocal next_shape
      next_shape = output.shape

    net.model.model[1].model[3].model[3].model[3].model[3].model[1].register_forward_hook(save_inner_shape)
    net.model.model[1].model[3].model[3].model[3].model[3].model[3].register_forward_hook(save_next_shape)

    net(net_input)
    self.assertEqual(inner_shape, (1, 512, 1, 1, 1))
    self.assertEqual(next_shape, (1, 512, 1, 2, 2))
  
  def test_default_UnetSkipConnectionBlock(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    net = UnetGenerator(1, 1, 5, norm_layer=norm_layer, use_dropout=True)

    net_input = torch.rand((1, 1, 32, 64, 64))

    inner_shape = None
    next_shape = None
    def save_inner_shape(_, __, output):
      nonlocal inner_shape
      inner_shape = output.shape
    def save_next_shape(_, __, output):
      nonlocal next_shape
      next_shape = output.shape

    net.model.model[1].model[3].model[3].model[3].model[1].register_forward_hook(save_inner_shape)
    net.model.model[1].model[3].model[3].model[3].model[3].register_forward_hook(save_next_shape)

    net(net_input)
    self.assertEqual(inner_shape, (1, 512, 1, 2, 2))
    self.assertEqual(next_shape, (1, 512, 2, 4, 4))


class NLayerDiscriminatorTest(unittest.TestCase):
  def test_original_discriminator(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    disc = NLayerDiscriminator(1, norm_layer=norm_layer)

    self.assertEqual(disc(torch.rand((1, 32, 64, 64))).shape, (1, 2, 6, 6))
    self.assertEqual(disc(torch.rand((1, 128, 128, 128))).shape, (1, 14, 14, 14))

  def test_condense_to_1_voxel(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    disc = NLayerDiscriminator(1, n_layers=6, norm_layer=norm_layer, disc_1x2x2_kernel_scale=True,
        disc_extra_xy_conv=True, disc_no_decrease_last_layers=True)
    self.assertEqual(disc(torch.rand((1, 32, 64, 64))).shape, (1, 1, 1, 1))

    disc = NLayerDiscriminator(1, n_layers=5, norm_layer=norm_layer, disc_1x2x2_kernel_scale=True,
        disc_extra_xy_conv=True, disc_no_decrease_last_layers=True)
    self.assertEqual(disc(torch.rand((1, 16, 32, 32))).shape, (1, 1, 1, 1))

    disc = NLayerDiscriminator(1, n_layers=6, norm_layer=norm_layer, disc_1x2x2_kernel_scale=False,
        disc_extra_xy_conv=True, disc_no_decrease_last_layers=True)
    self.assertEqual(disc(torch.rand((1, 32, 64, 64))).shape, (1, 1, 1, 1))

    disc = NLayerDiscriminator(1, n_layers=5, norm_layer=norm_layer, disc_1x2x2_kernel_scale=False,
        disc_extra_xy_conv=False, disc_no_decrease_last_layers=True)
    self.assertEqual(disc(torch.rand((1, 32, 32, 32))).shape, (1, 1, 1, 1))


class ScalerTest(unittest.TestCase):
    def test_scale_tanh(self):
        scaler = Scaler(-1, 1, 0, 255)
        image = (torch.rand(200) - 0.5) * 2
        converted = scaler(image)

        torch.testing.assert_close(converted, (image + 1)* 127.5)

    def test_scale_sigmoid(self):
        scaler = Scaler(0, 1, 0, 255)
        image = torch.rand(200)
        converted = scaler(image)

        torch.testing.assert_close(converted, image * 255)


class ProgressiveDiscriminatorTest(unittest.TestCase):
    def test_module_list_len(self):
        disc = ProgressiveDiscriminator(1, 1, 5, 10, 0.5)
        self.assertEqual(len(disc.prog_blocks), 4)
        self.assertEqual(len(disc.output_layers), 5)

        disc = ProgressiveDiscriminator(1, 2, 7, 10, 0.5)
        self.assertEqual(len(disc.prog_blocks), 6)
        self.assertEqual(len(disc.output_layers), 7)

        disc = ProgressiveDiscriminator(2, 1, 4, 100, 0.2)
        self.assertEqual(len(disc.prog_blocks), 3)
        self.assertEqual(len(disc.output_layers), 4)

    def test_growing(self):
        disc = ProgressiveDiscriminator(1, 1, 5, 10, 0.5)

        self.assertEqual(disc.n_layers, 1)
        self.assertEqual(disc.alpha, 1)
        self.assertEqual(disc.data_till_new_layer, 5)

        disc.batch_processed(1)
        self.assertEqual(disc.n_layers, 1)
        self.assertEqual(disc.alpha, 1)
        self.assertEqual(disc.data_till_new_layer, 4)

        for i in range(3):
            disc.batch_processed(1)
            self.assertEqual(disc.data_till_new_layer, 3-i)

        disc.batch_processed(1)
        self.assertEqual(disc.n_layers, 2)
        self.assertEqual(disc.alpha, 1e-5)
        self.assertEqual(disc.data_till_new_layer, 10)

        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1e-5 + 0.2)
        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1e-5 + 0.4)
        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1e-5 + 0.6)
        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1e-5 + 0.8)
        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1)
        disc.batch_processed(1)
        self.assertAlmostEqual(disc.alpha, 1)

        for _ in range(4):
            disc.batch_processed(1)

        self.assertEqual(disc.n_layers, 3)
        self.assertEqual(disc.alpha, 1e-5)
        self.assertEqual(disc.data_till_new_layer, 10)

    def test_forward(self):
        # This method tests which blocks of the discriminator were
        # called during a forward pass. This is done via pytorch hooks.
        disc = ProgressiveDiscriminator(1, 1, 5, 10, 0.5)

        prog_block_called = torch.Tensor([False] * len(disc.prog_blocks))
        out_layer_called = torch.Tensor([False] * len(disc.output_layers))

        def register_block(i):
            def hook(module, input, output):
                prog_block_called[i] = True
            disc.prog_blocks[i].register_forward_hook(hook)

        def register_out(i):
            def hook(module, input, output):
                out_layer_called[i] = True
            disc.output_layers[i].register_forward_hook(hook)

        for i in range(len(disc.prog_blocks)):
            register_block(i)
        for i in range(len(disc.output_layers)):
            register_out(i)

        def clear_calls():
            prog_block_called[:] = False
            out_layer_called[:] = False

        def assert_block_called_until(n):
            if n < 0:
                self.assertFalse(any(prog_block_called))
                return
            self.assertTrue(all(prog_block_called[:n+1]))
            self.assertFalse(any(prog_block_called[n+1:]))

        def assert_only_outs_called(*indices):
            self.assertTrue(all(out_layer_called[list(indices)]))
            false_incides = list(set(range(len(out_layer_called))) - set(indices))
            self.assertFalse(any(out_layer_called[false_incides]))

        disc(torch.rand((1, 1, 32, 64, 64)))
        assert_block_called_until(-1)
        assert_only_outs_called(0)
        clear_calls()

        disc.n_layers = 2

        disc(torch.rand((1, 1, 32, 64, 64)))
        assert_block_called_until(0)
        assert_only_outs_called(0, 1)
        clear_calls()

        disc.n_layers = 3

        disc(torch.rand((1, 1, 32, 64, 64)))
        assert_block_called_until(1)
        assert_only_outs_called(1, 2)
        clear_calls()

        disc.n_layers = 5

        disc(torch.rand((1, 1, 32, 64, 64)))
        assert_block_called_until(3)
        assert_only_outs_called(3, 4)
