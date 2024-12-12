#!/usr/bin/env python
import sys
import os
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

# Avoids warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

import ScalableLib.classifier.Multiband as multiband


# In[2]:


# To see if the system recognises the GPU
device = 0
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(devices[device], 'GPU')
tf.config.experimental.set_memory_growth(device=devices[device], enable=True)

device_name = tf.config.experimental.get_device_details(devices[device])['device_name']
print("Using {}".format(device_name))


# Find the different folds and train a model using the stored data.

# In[3]:


survey = 'ZTF'


# In[4]:


path = os.path.join('../../02_CreateRecords/', survey, 'Folds/Fold_*',)
folds = glob(path)
folds.sort()
print(folds)


# Create folder results

# In[5]:


if not os.path.exists('./Results'):
    os.mkdir('./Results')


# Define the arguments for all the models.

# In[6]:


train_args = {
            'hidden_size_bands':[128,128, 128],
            'hidden_size_central':[128, 128],
            'fc_layers_bands':[128,128,128],
            'fc_layers_central':[128,128,128], # Neurons of each layer
            'regression_size':[128, 128],#each element is a layer with that size.
            'buffer_size':10000,
            'epochs':1000,
            'num_threads':7,
            'batch_size':256,
            'dropout':0.30,
            'lr':[[5e-3]*2, 2.5e-3], # [[band1, band2], central]
            'val_steps':50,
            'max_to_keep':0, # Not Used 
            'steps_wait':500, 
            'use_class_weights':True,# Not Used as intended, for initialization
            'mode' : 'classifier+regression',
            }
loss_weights = {'Class':300.0, 'T_eff':20.0,'Radius':1e0}

callbacks_args = {'patience': 20,
                  'mode':'max',
                  'restore_best_weights':True,
                  'min_delta': 0.001
                 }
train_args_specific={
                    'phys_params': ['T_eff', 'Radius'],
                    'use_output_bands' : True,  # Working
                    'use_output_central' : False, # Not used
                    'use_common_layers' : False, # NOT Working
                    'bidirectional_central' : False,# Working
                    'bidirectional_band' : False,# Not Working
                    'layer_norm_params' : None, # Used to normalyze common layers
                    'use_gated_common' : False, # Working
                    'l1':0.0,
                    'l2':0.0,  
                    'N_skip': 3, # Cannot be greater than the number of timesteps
                    'use_raw_input_central': False,
                    'train_steps_central' : 2,
                    'print_report' : True,
                    'loss_weights_central' : loss_weights,
                    'callbacks_args':callbacks_args
                    }



# In[10]:


for fold in folds[1:5]:
    tf.keras.backend.clear_session()
    # Set the fold path
    base_dir = fold+'/'
    
    # Set the save path for this fold. Create folder if needed
    path_results_fold = fold.replace('../../02_CreateRecords/'+survey+'/', './').replace('/Folds/', '/Results/')
    if not os.path.exists(path_results_fold):
        os.mkdir(path_results_fold)
    
    train_args_specific['save_dir'] = path_results_fold
    train_args_specific['metadata_pre_path'] = base_dir+'metadata_preprocess.json'
    train_args_specific['path_scalers'] =  os.path.join(fold,'scalers')
                                 
    # Define the train args
    train_args = {**train_args, **train_args_specific}


    train_files = base_dir+'train/*.tfrecord'
    val_files = base_dir+'val/*.tfrecord'
    test_files = base_dir+'test/*.tfrecord' 
    
    new = multiband.Network()
    new.train(train_args, train_files, val_files, test_files)    
    new.train_loop()



