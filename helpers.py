import sys
import helpers.dataset_previewer as dataset_previewer

if len(sys.argv) < 2:
  print('You must provide the name of the helper which shall be invoked!')
  exit(1)

helper = sys.argv[1]

if helper == 'dataset_previewer':
  del sys.argv[1]
  dataset_previewer.main()
else:
  print('invalid helper provided')