import functools
import unittest
import torch
import torch.nn as nn

from models.networks_3d import ResnetGenerator, get_post_transform, UnetGenerator

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


class PostTransformTest(unittest.TestCase):
  def test_identity(self):
    transform = get_post_transform('identity')

    t = torch.tensor([-0.2, 0.5, 2, 8.41, -13.20])
    transformed_t = transform(t)
    self.assertTrue((t == transformed_t).all())

    t = torch.rand(100)
    transformed_t = transform(t)
    self.assertTrue((t == transformed_t).all())

  def test_map_binary(self):
    transform = get_post_transform('map_binary', -14, 14)

    t = torch.tensor([-0.2, 0.5, 2, 8.41, -13.20])
    transformed_t = transform(t)
    self.assertTrue((transformed_t == torch.tensor([-14, 14, 14, 14, -14])).all())
    self.assertFalse((transformed_t == t).all())

    t = torch.rand(100)
    transformed_t = transform(t)
    self.assertTrue((transformed_t == torch.sign(t) * 14).all())

    transform = get_post_transform('map_binary', 4, 10)

    t = torch.tensor([4, 6, 5.3, 8.3, 10, 8.9, 7, 6.99])
    transformed_t = transform(t)
    self.assertTrue((transformed_t == torch.tensor([4, 4, 4, 10, 10, 10, 10, 4])).all())


class UnetGenerator3dTest(unittest.TestCase):
  def test_correct_shape(self):
    norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    net = UnetGenerator(1, 1, 5, norm_layer=norm_layer, use_dropout=True)

    net_input = torch.rand((1, 32, 64, 64))
    net_output = net(net_input)
    self.assertEqual(net_output.shape, (1, 32, 64, 64))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 32, 32, 32))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 32, 32, 32))
    self.assertEqual(net_output.dtype, torch.float32)
    self.assertLessEqual(net_output.max(), 1)
    self.assertGreaterEqual(net_output.min(), -1)

    net_input = torch.rand((1, 64, 128, 64))
    net_output = net(net_input)
    self.assertEqual(net(net_input).shape, (1, 64, 128, 64))
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

    net_input = torch.rand((1, 32, 64, 64))
    net_output = net(net_input)
    self.assertEqual(net_output.shape, (1, 32, 64, 64))
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

    net_input = torch.rand((1, 32, 64, 64))

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