import os
import sys
from glob import glob

sys.path.append('../src/')
import base.Load_Process as Load_Process

survey = 'Gaia'
path_folds = os.path.join('./Folds/*')

# Get the folder paths
folds = glob(path_folds)
# Get the test file path
test_path = [i for i in folds if 'test' in i][0]
# Remove the test path from the list
folds.remove(test_path)
# Sort to work from the first fold onwards
folds.sort()

# For each fold, process the information and store it.

# The **P** object is defined with its keyword arguments.
#
# Then the kwargs are defined to be used as inputs of the prepare method.
kwargs1 = dict(max_l=1000, min_l=500, min_n=10, max_n=1000, w=2, s=1, njobs=7, n_bands=2)


def define_p(kwargs):
    P = Load_Process.prepData(**kwargs)
    return P


P = define_p(kwargs1)

kwargs2 = {
    'dataset_header': ['ID', 'Path', 'Class', 'N', 'N_b', 'N_r'],
    'lc_parameters': {'sep': ',', 'header': 0, 'na_filter': True, 'engine': 'c'},
    'params_phys': ['T_eff', 'Lum', 'Mass', 'rho', 'Radius', 'logg'],
    'params_phys_est': [],
}

# Create the container folder
if not os.path.exists('./Folds'):
    os.mkdir('./Folds')

for fold in folds:
    train_files = glob(os.path.join(fold, '*'))
    train_path = [i for i in train_files if 'train' in i][0]
    val_path = [i for i in train_files if 'val' in i][0]

    path_fold = fold.replace('./../01_CreateFolds/', './') + '/'
    if not os.path.exists(path_fold):
        os.mkdir(path_fold)

    print(path_fold)

    kwargs2['file_train'] = train_path
    kwargs2['file_val'] = val_path
    kwargs2['file_test'] = test_path
    kwargs2['save_dir'] = path_fold
    P.prepare(**kwargs2)
