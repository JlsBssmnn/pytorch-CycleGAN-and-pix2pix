import unittest
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import ToTensor


toTensor = transforms.ToTensor()


class BaseDatasetTest(unittest.TestCase):
  def test_to_tensor_2d(self):
    test_count = 10
    sample_shape = (128, 128)
    for _ in range(test_count):
      array = np.random.randint(0, 255, sample_shape, np.uint8)
      self.assertTrue((ToTensor(array) == toTensor(array)).all())

  def test_to_tensor_3d(self):
    test_count = 10
    sample_shape = (64, 128, 128)
    for _ in range(test_count):
      array = np.random.randint(0, 255, sample_shape, np.uint8)
      tensor = ToTensor(array)
      self.assertGreaterEqual(tensor.min(), 0)
      self.assertLessEqual(tensor.max(), 255)
