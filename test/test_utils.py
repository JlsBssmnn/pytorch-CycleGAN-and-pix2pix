import os
import shutil
tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'tmp'))

def clear_tmp():
  for file in os.listdir(tmp_dir):
    path = os.path.join(tmp_dir, file)
    if os.path.isdir(path):
      shutil.rmtree(path)
    else:
      os.remove(path)