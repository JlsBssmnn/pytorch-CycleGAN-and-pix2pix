import unittest

import numpy as np
from data.h5_folder import get_datasets
import h5py
import os

from test.test_utils import tmp_dir, clear_tmp

class TestH5Folder(unittest.TestCase):

  def test_get_all_datasets(self):
    with h5py.File(os.path.join(tmp_dir, 'test.h5'), 'w') as f:
      f.create_dataset('dataset1', data=np.array([1]))
      f.create_dataset('group1/dataset2', data=np.array([2]))
      f.create_dataset('group1/subgroup1/dataset3', data=np.array([3]))
      f.create_group('group2')
      f.create_group('group3/subgroup1')
      g = f.create_group('group4/subgroup1/subgroup2')
      g.create_dataset('deep_dataset', data=np.array([4]))

    datasets = get_datasets(os.path.join(tmp_dir, 'test.h5'))
    self.assertEqual(len(datasets), 4)
    self.assertEqual(set([array[0] for array in datasets]),
      set([1, 2, 3, 4]))

  def test_get_filtered_datasets(self):
    with h5py.File(os.path.join(tmp_dir, 'test.h5'), 'w') as f:
      f.create_dataset('dataset1', data=np.array([1]))
      f.create_dataset('group1/dataset2', data=np.array([2]))
      f.create_dataset('group1/subgroup1/dataset3', data=np.array([3]))
      f.create_group('group2')
      f.create_group('group3/subgroup1')
      g = f.create_group('group4/subgroup1/subgroup2')
      g.create_dataset('deep_dataset', data=np.array([4]))

    datasets = get_datasets(os.path.join(tmp_dir, 'test.h5'), ['dataset1', 'deep_dataset'])
    self.assertEqual(len(datasets), 2)
    self.assertEqual(set([array[0] for array in datasets]),
      set([1, 4]))

  def tearDown(self) -> None:
    clear_tmp()
    return super().tearDown()