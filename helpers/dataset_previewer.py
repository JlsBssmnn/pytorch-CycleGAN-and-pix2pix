from data.unaligned_3d_dataset import Unaligned3dDataset
from options.train_options import TrainOptions

def main():
  opt = TrainOptions().parse()
  dataset = Unaligned3dDataset(opt)

  print('the size of the dataset is:', dataset.len)