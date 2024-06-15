import sys
import tensorflow as tf
import os
from glob import glob

# sys.path.append('../')
# sys.path.append('./../src')

import ScalableLib.classifier.Multiband as multiband

# To see if the system recognises the GPU
device = 1
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(devices[device], 'GPU')
tf.config.experimental.set_memory_growth(device=devices[device], enable=True)

survey = 'Gaia'
path = './Folds/Fold_*/'
folds = glob(path)
folds.sort()

if not os.path.exists('./Results'):
    os.mkdir('./Results')

train_args = {
    'hidden_size_bands': [16, 16, 16],
    'hidden_size_central': [16, 16],
    'fc_layers_bands': [16, 16, 16],
    'fc_layers_central': [16, 16, 16],  # Neurons of each layer
    'regression_size': [16, 16],  # each element is a layer with that size.
    'buffer_size': 10000,
    'epochs': 2,
    'num_threads': 7,
    'batch_size': 4096,
    'dropout': 0.40,
    'lr': [[1e-3] * 2, 1e-3],  # [[band1, band2], central]
    'val_steps': 50,
    'max_to_keep': 0,  # Not Used
    'steps_wait': 0,
    'use_class_weights': False,  # Not Used
    'mode': 'classifier+regression'
}
loss_weights = {'Class': 300.0, 'T_eff': 20.0, 'Radius': 1e0}

callbacks_args = {'patience': 20,
                  'mode': 'max',
                  'restore_best_weights': True,
                  'min_delta': 0.001
                  }
train_args_specific = {
    'phys_params': ['T_eff', 'Radius'],
    'use_output_bands': True,  # Working
    'use_output_central': False,  # Not used
    'use_common_layers': False,  # NOT Working
    'bidirectional_central': False,  # Working
    'bidirectional_band': False,  # Not Working
    'layer_norm_params': None,  # Used to normalize common layers
    'use_gated_common': False,  # Working
    'l1': 0.0,
    'l2': 0.0,
    'N_skip': 8,  # Cannot be greater than the number of time steps
    'use_raw_input_central': True,
    'train_steps_central': 2,
    'print_report': True,
    'loss_weights_central': loss_weights,
    'callbacks_args': callbacks_args
}

for fold in folds:
    tf.keras.backend.clear_session()
    # Set the fold path
    base_dir = fold

    # Set the save path for this fold. Create folder if needed
    # path_results_fold = fold.replace('../notebooks/02_CreateRecords', '.').replace('/Folds/', '/Results/')
    path_results_fold = fold.replace('Folds', 'Results')
    if os.path.exists(path_results_fold):
        pass
    else:
        os.mkdir(path_results_fold)

    train_args_specific['save_dir'] = path_results_fold
    train_args_specific['metadata_pre_path'] = base_dir + 'metadata_preprocess.json'
    train_args_specific['path_scalers'] = base_dir + 'scalers.pkl'
    # Define the train args
    train_args = {**train_args, **train_args_specific}

    train_files = base_dir + 'train/*.tfrecord'
    val_files = base_dir + 'val/*.tfrecord'
    test_files = base_dir + 'test/*.tfrecord'

    new = multiband.Network()
    new.train(train_args, train_files, val_files, test_files)
    new.train_loop()
