import sys
from helpers import config_analyser
import helpers.dataset_previewer as dataset_previewer

if len(sys.argv) < 2:
    print('You must provide the name of the helper which shall be invoked!')
    exit(1)

helper = sys.argv[1]
del sys.argv[1]

if helper == 'dataset_previewer':
    dataset_previewer.main()
elif helper == 'config_analyser':
    config_analyser.main()
else:
    print('invalid helper provided')
