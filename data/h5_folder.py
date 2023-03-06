import h5py
import numpy as np

def get_datasets(file, datasets=None, exclude=None, dtype=None):
  """
  Extracts datasets from the given h5 file path and returns a list of numpy
  arrays that contain these datasets. The datasets can be filtered with the
  `datasets` parameter. If specified, only datasets whose names are in the
  `datasets` list are returned. If `datasets` is not specified, all datasets
  are returned.

  Parameters:
  -------
  file: The filename of the h5 file
  datasets: Which datasets to extract (default: everything)
  exclude: A list of datasets which shall not be extracted (default: None)
  """
  dataset_names = []
  def get_dataset(name, node):
    nonlocal dataset_names
    if isinstance(node, h5py.Dataset):
      dataset_names.append(name)

  with h5py.File(file) as f:
    f.visititems(get_dataset)
    data = []
    for dataset in dataset_names:
      d = f[dataset]
      dataset_name = d.name[d.name.rfind('/')+1:]
      if (exclude is None or dataset_name not in exclude) and (datasets is None or dataset_name in datasets):
        if dtype is None:
          data.append(np.asarray(d))
        else:
          data.append(np.asarray(d, dtype=dtype))
    return data
