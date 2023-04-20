from operator import itemgetter
import unittest
import h5py
import os
import numpy as np
import numpy.testing as npt
import torch
from data.unaligned_3d_dataset import Unaligned3dDataset

from test.test_utils import tmp_dir, clear_tmp

def create_train_data(data=None):
  os.mkdir(os.path.join(tmp_dir, 'trainA'))
  os.mkdir(os.path.join(tmp_dir, 'trainB'))
  if data is None:
    data = np.arange(4000).reshape((1, 10, 20, 20))

  with h5py.File(os.path.join(tmp_dir, 'trainA', 'test.h5'), 'w') as f:
    f.create_dataset('dataset1', data=data)

  with h5py.File(os.path.join(tmp_dir, 'trainB', 'test.h5'), 'w') as f:
    f.create_dataset('dataset1', data=data)

class TestOptions:
  border_offset_A = [0, 0, 0]
  border_offset_B = [0, 0, 0]
  dataroot = tmp_dir
  phase = 'train'
  datasetA_file = 'test.h5'
  datasetB_file = 'test.h5'
  direction = 'AtoB'
  input_nc = 1
  output_nc = 1
  no_normalization = True
  sample_size = [3, 10, 7]
  datasetA_names = None
  datasetB_names = None
  datasetA_mask = None
  datasetB_mask = None
  dataset_length = 'max'
  serial_batches = True
  dataset_stride: None | tuple[int, int, int] = None
  datasetA_mask: None | str = None
  datasetB_mask: None | str = None

  def __init__(self, sample_size=None):
    if sample_size is not None:
      self.sample_size = sample_size

base_sample = torch.tensor([
  [[0, 1, 2, 3],
    [20, 21, 22, 23],
    [40, 41, 42, 43]],
  [[400, 401, 402, 403],
    [420, 421, 422, 423],
    [440, 441, 442, 443]],
])


class Unaligned3dDatasetTest(unittest.TestCase):
  def test_instantiation(self):
    create_train_data()

    dataset = Unaligned3dDataset(TestOptions())
    self.assertEqual(len(dataset), 12)
    self.assertEqual(dataset[0]['A'].shape, (1, 3, 10, 7))

    with self.assertRaises(AssertionError):
      dataset[12]
      dataset[13]
      dataset[14]

  def test_wrong_dimensionality(self):
    create_train_data(np.arange(4000).reshape((10, 20, 20)))

    with self.assertRaises(AssertionError):
      Unaligned3dDataset(TestOptions())


  def test_samples(self):
    create_train_data()

    dataset = Unaligned3dDataset(TestOptions([2, 3, 4]))
    self.assertEqual(len(dataset), 5*6*5)
    self.assertEqual(dataset[0]['A'].shape, (1, 2, 3, 4))
    self.assertEqual(dataset[0]['A'][0, 0, 0, 0], 0)
    self.assertTrue((dataset[0]['A'][0] == (base_sample + 0)).all())

    self.assertEqual(dataset[1]['A'][0, 0, 0, 0], 4)
    self.assertTrue((dataset[1]['A'][0] == (base_sample + 4)).all())

    self.assertEqual(dataset[2]['A'][0, 0, 0, 0], 8)
    self.assertTrue((dataset[2]['A'][0] == (base_sample + 8)).all())

    self.assertEqual(dataset[6]['A'][0, 0, 0, 0], 64)
    self.assertTrue((dataset[6]['A'][0] == (base_sample + 64)).all())

    self.assertEqual(dataset[31]['A'][0, 0, 0, 0], 804)
    self.assertTrue((dataset[31]['A'][0] == (base_sample + 804)).all())


  def test_samples_with_mask(self):
    create_train_data()

    with h5py.File(os.path.join(tmp_dir, 'trainA', 'test.h5'), 'a') as f:
      mask = np.ones((10, 20, 20), dtype=bool)
      mask[0, 5:15, 5:15] = False
      mask[1:8, 0:5, 0:3] = False
      mask[7:10, 16:, 13:18] = False
      f.create_dataset('mask', data=mask)
    with h5py.File(os.path.join(tmp_dir, 'trainB', 'test.h5'), 'a') as f:
      mask = np.ones((10, 20, 20), dtype=bool)
      mask[0, 5:15, 5:15] = False
      mask[1:8, 0:5, 0:3] = False
      mask[7:10, 16:, 13:18] = False
      f.create_dataset('mask', data=mask)

    options = TestOptions([2, 3, 4])
    options.datasetA_mask = 'mask'
    options.datasetB_mask = 'mask'
    dataset = Unaligned3dDataset(options)

    # Number of samples in each z-slice
    # slice 0 <= z < 2: 16
    # slice 2 <= z < 4: 28
    # slice 4 <= z < 6: 28
    # slice 6 <= z < 8: 26
    # slice 8 <= z < 10: 28

    self.assertEqual(len(dataset), 126)
    self.assertEqual(dataset[0]['A'].shape, (1, 2, 3, 4))
    self.assertEqual(dataset[0]['A'][0, 0, 0, 0], 4)
    self.assertTrue((dataset[0]['A'][0] == (base_sample + 4)).all())

    self.assertEqual(dataset[4]['A'][0, 0, 0, 0], 76)
    self.assertTrue((dataset[4]['A'][0] == (base_sample + 76)).all())

    self.assertEqual(dataset[5]['A'][0, 0, 0, 0], 120)
    self.assertTrue((dataset[5]['A'][0] == (base_sample + 120)).all())

    self.assertEqual(dataset[6]['A'][0, 0, 0, 0], 136)
    self.assertTrue((dataset[6]['A'][0] == (base_sample + 136)).all())

    self.assertEqual(dataset[7]['A'][0, 0, 0, 0], 180)
    self.assertTrue((dataset[7]['A'][0] == (base_sample + 180)).all())


    # z = 2-3
    self.assertEqual(dataset[16]['A'][0, 0, 0, 0], 804)
    self.assertTrue((dataset[16]['A'][0] == (base_sample + 804)).all())

    self.assertEqual(dataset[17]['A'][0, 0, 0, 0], 808)
    self.assertTrue((dataset[17]['A'][0] == (base_sample + 808)).all())

    self.assertEqual(dataset[20]['A'][0, 0, 0, 0], 864)
    self.assertTrue((dataset[20]['A'][0] == (base_sample + 864)).all())

    self.assertEqual(dataset[24]['A'][0, 0, 0, 0], 920)
    self.assertTrue((dataset[24]['A'][0] == (base_sample + 920)).all())


    # z = 4-5
    self.assertEqual(dataset[44]['A'][0, 0, 0, 0], 1604)
    self.assertTrue((dataset[44]['A'][0] == (base_sample + 1604)).all())

    self.assertEqual(dataset[71]['A'][0, 0, 0, 0], 1916)
    self.assertTrue((dataset[71]['A'][0] == (base_sample + 1916)).all())


    # z = 6-7
    self.assertEqual(dataset[72]['A'][0, 0, 0, 0], 2404)
    self.assertTrue((dataset[72]['A'][0] == (base_sample + 2404)).all())

    self.assertEqual(dataset[97]['A'][0, 0, 0, 0], 2708)
    self.assertTrue((dataset[97]['A'][0] == (base_sample + 2708)).all())


    # z = 8-9
    self.assertEqual(dataset[98]['A'][0, 0, 0, 0], 3200)
    self.assertTrue((dataset[98]['A'][0] == (base_sample + 3200)).all())

    self.assertEqual(dataset[125]['A'][0, 0, 0, 0], 3508)
    self.assertTrue((dataset[125]['A'][0] == (base_sample + 3508)).all())

  def test_different_dataset_sizes(self):
    os.mkdir(os.path.join(tmp_dir, 'trainA'))
    os.mkdir(os.path.join(tmp_dir, 'trainB'))
    with h5py.File(os.path.join(tmp_dir, 'trainA', 'test.h5'), 'w') as f:
      f.create_dataset('dataset1', data=np.arange(4000).reshape((1, 10, 20, 20)))
    with h5py.File(os.path.join(tmp_dir, 'trainB', 'test.h5'), 'w') as f:
      f.create_dataset('dataset1', data=np.arange(6000).reshape((1, 15, 20, 20)))

    dataset = Unaligned3dDataset(TestOptions([2, 3, 4]))
    self.assertEqual(len(dataset), 7*6*5)

    self.assertEqual(dataset[149]['A'][0, 0, 0, 0], 3516)
    self.assertEqual(dataset[149]['B'][0, 0, 0, 0], 3516)

    self.assertEqual(dataset[150]['A'][0, 0, 0, 0], 0)
    self.assertEqual(dataset[150]['B'][0, 0, 0, 0], 4000)

    self.assertEqual(dataset[151]['A'][0, 0, 0, 0], 4)
    self.assertEqual(dataset[151]['B'][0, 0, 0, 0], 4004)

    self.assertEqual(dataset[179]['A'][0, 0, 0, 0], 316)
    self.assertEqual(dataset[179]['B'][0, 0, 0, 0], 4316)

    self.assertEqual(dataset[180]['A'][0, 0, 0, 0], 800)
    self.assertEqual(dataset[180]['B'][0, 0, 0, 0], 4800)

    self.assertEqual(dataset[209]['A'][0, 0, 0, 0], 1116)
    self.assertEqual(dataset[209]['B'][0, 0, 0, 0], 5116)

  def test_normalization(self):
    data = np.arange(4000).reshape((1, 10, 20, 20))
    data %= 256
    create_train_data(data.astype(np.uint8))
    opt = TestOptions([2, 3, 4])
    opt.no_normalization = False
    dataset = Unaligned3dDataset(opt)

    for i in range(100):
      A, B = itemgetter('A', 'B')(dataset[i])
      self.assertLessEqual(A.max(), 1)
      self.assertGreaterEqual(A.min(), -1)
      self.assertLessEqual(B.max(), 1)
      self.assertGreaterEqual(B.min(), -1)

  def test_no_serial_batches(self):
    create_train_data(np.arange(64).reshape(1, 4, 4, 4))
    opt = TestOptions([2, 2, 2])
    opt.serial_batches = False
    dataset = Unaligned3dDataset(opt)
    seen_samples = set()

    for _ in range(200):
      seen_samples.add(tuple(dataset[0]['B'].numpy().flatten()))
      if len(seen_samples) == 8:
          break
    self.assertEqual(len(seen_samples), 8)

  def test_smaller_stride(self):
    create_train_data(np.arange(64).reshape(1, 4, 4, 4))
    opt = TestOptions([2, 2, 2])
    opt.serial_batches = False
    opt.dataset_stride = (1, 1, 1)
    dataset = Unaligned3dDataset(opt)

    self.assertEqual(len(dataset), 27)
    npt.assert_array_equal(dataset[0]['A'], np.array([[[[0, 1], [4, 5]], [[16, 17], [20, 21]]]]))
    npt.assert_array_equal(dataset[1]['A'], np.array([[[[1, 2], [5, 6]], [[17, 18], [21, 22]]]]))
    npt.assert_array_equal(dataset[2]['A'], np.array([[[[2, 3], [6, 7]], [[18, 19], [22, 23]]]]))

    self.assertEqual(dataset[3]['A'][0, 0, 0, 0], 4)
    self.assertEqual(dataset[4]['A'][0, 0, 0, 0], 5)
    self.assertEqual(dataset[5]['A'][0, 0, 0, 0], 6)

    self.assertEqual(dataset[9]['A'][0, 0, 0, 0], 16)
    self.assertEqual(dataset[10]['A'][0, 0, 0, 0], 17)

    self.assertEqual(dataset[18]['A'][0, 0, 0, 0], 32)
    self.assertEqual(dataset[26]['A'][0, 0, 0, 0], 42)

  def test_smaller_stride_with_mask(self):
    create_train_data(np.arange(512).reshape(1, 8, 8, 8))
    with h5py.File(os.path.join(tmp_dir, 'trainA', 'test.h5'), 'a') as f:
      mask = np.ones((10, 20, 20), dtype=bool)
      mask[:, :, -1] = False # should have no effect because x stride is 3
      mask[4, 0:4, 0:4] = False
      mask[7, 7, 6] = False
      f.create_dataset('mask', data=mask)

    opt = TestOptions([4, 4, 4])
    opt.datasetA_mask = 'mask'
    opt.dataset_stride = (2, 1, 3)
    opt.dataset_length = 'min'
    dataset = Unaligned3dDataset(opt)

    self.assertEqual(len(dataset), 3*5*2 - 16 - 1)
    npt.assert_array_equal(dataset[0]['A'], np.array([[[[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19], [24, 25, 26, 27]],
                                                       [[64, 65, 66, 67], [72, 73, 74, 75], [80, 81, 82, 83], [88, 89, 90, 91]],
                                                       [[128, 129, 130, 131], [136, 137, 138, 139], [144, 145, 146, 147], [152, 153, 154, 155]],
                                                       [[192, 193, 194, 195], [200, 201, 202, 203], [208, 209, 210, 211], [216, 217, 218, 219]]]]))
    self.assertEqual(dataset[1]['A'][0, 0, 0, 0], 3)
    self.assertEqual(dataset[2]['A'][0, 0, 0, 0], 8)
    self.assertEqual(dataset[3]['A'][0, 0, 0, 0], 11)
    self.assertEqual(dataset[9]['A'][0, 0, 0, 0], 35)
    self.assertEqual(dataset[10]['A'][0, 0, 0, 0], 160)
    self.assertEqual(dataset[11]['A'][0, 0, 0, 0], 163)
    self.assertEqual(dataset[12]['A'][0, 0, 0, 0], 288)
    
  def setUp(self) -> None:
    clear_tmp()
    return super().setUp()

  def tearDown(self) -> None:
    clear_tmp()
    return super().tearDown()
