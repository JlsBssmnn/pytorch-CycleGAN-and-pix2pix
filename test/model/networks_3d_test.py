import functools
import unittest
import torch
import torch.nn as nn

from models.networks_3d import ResnetGenerator
from models.networks import ResnetGenerator as ResnetGenerator2d

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