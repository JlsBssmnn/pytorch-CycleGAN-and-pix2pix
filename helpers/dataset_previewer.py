from data.unaligned_3d_dataset import Unaligned3dDataset
from options.train_options import TrainOptions
import sys

from util.get_options import get_training_options

def main():
  opt = get_training_options()
  dataset = Unaligned3dDataset(opt)

  print('the size of the dataset is:', dataset.len)