import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import BaseClass.Parser_02 as Parser
from datetime import datetime
from BaseClass.Modified_GRUCell import ModGRUCell
from BaseClass.CustomDense import custom_Dense
# import tensorflow_addons as tfa
class Network():
    def __init__(self):
        tf.keras.backend.clear_session()

    def load_preprocess_data(self, path):
        with open(path) as f:
            metadata = json.load(f)
            self.w = int(metadata['w'])
            # self.info = list(metadata['info'])
            # Multiply w by the number of columns included
            self.w = int(2*self.w)
            self.s = int(metadata['s'])
            self.n_bands = int(metadata['Number of bands'])
            self.max_L = int(metadata['Max per class'])
            self.min_L = int(metadata['Min per class'])
            self.max_N = int(metadata['Max points per lc'])
            self.min_N = int(metadata['Min points per lc'])
            self.num_classes= int(metadata['Number of classes'])

            # self.size_train = int(metadata['Classes Info']['Train set']['Total'])
            # self.size_test = int(metadata['Classes Info']['Test set']['Total'])
            # self.size_val = int(metadata['Classes Info']['Val set']['Total'])
            trans = metadata['Classes Info']['Keys']


            keys = [int(k) for k in trans.keys()]
            self.trans = dict(zip(keys, trans.values()))
            self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))
            # Get numbers per class
            self.element_class = metadata['Classes Info']['Train set']
            self.element_class = {self.trans_inv[i]:int(self.element_class[i]) for i in self.element_class.keys()}

    def set_train_settings(self, train_args):

        self.load_preprocess_data(train_args['metadata_pre_path'])
        self.physical_params = train_args['phys_params']

        self.size_hidden_bands = train_args['hidden_size_bands']
        self.size_hidden_central = train_args['hidden_size_central']

        self.rnn_layers_bands = len(self.size_hidden_bands)
        self.rnn_layers_central = len(self.size_hidden_central)

        self.fc_layers_bands = train_args['fc_layers_bands']
        self.fc_layers_central = train_args['fc_layers_central']

        self.regression_size = train_args['regression_size']

        self.buffer_size = train_args['buffer_size']
        self.epochs = train_args['epochs']
        self.num_cores = train_args['num_threads']
        self.batch_size = train_args['batch_size']
        self.dropout = train_args['dropout']
        self.lr_bands = train_args['lr'][0]
        self.lr_central = train_args['lr'][1]
        self.val_steps = train_args['val_steps']
        self.max_to_keep = train_args['max_to_keep']

        self.use_output_layers_bands = train_args['use_output_bands']
        self.use_output_layers_central = train_args['use_output_central']

        self.use_common_layers = train_args['use_common_layers']
        self.use_gated_common = train_args['use_gated_common']


        self.layer_norm_params = train_args['layer_norm_params']

        self.steps_wait = train_args['steps_wait']
        self.l1 = train_args['l1']
        self.l2 = train_args['l2']
        self.bidirectional_central = train_args['bidirectional_central']
        self.bidirectional_band = train_args['bidirectional_band']

        # Wether to use class weights or not
        self.use_class_weights = train_args['use_class_weights']
        if self.use_class_weights:
            total = np.sum([self.element_class[i] for i in self.element_class])/self.num_classes
            self.class_weights = {i:total/self.element_class[i] for i in self.element_class }
            self.numpy_weights = np.array([self.class_weights[i] for i in range(self.num_classes)])
            self.vector_weights = tf.constant(self.numpy_weights, dtype=tf.float32)
        else:
            self.class_weights = None
            self.numpy_weights = np.array([1.0 for i in range(self.num_classes)])
            self.vector_weights = tf.constant([1.0 for i in range(self.num_classes)], dtype=tf.float32)

        # Use current time to identify models
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        log_folder = os.path.join(train_args['save_dir'],'Logs', current_time)
        self.log_folder_train = os.path.join(log_folder ,'train')
        self.log_folder_val = os.path.join(log_folder ,'val')


        self.mode = train_args['mode']

        self.model_dir = os.path.join(train_args['save_dir'],'Models', current_time)

        self.N_skip = train_args['N_skip']
        self.use_raw_input_central = train_args['use_raw_input_central']
        self.train_steps_central = train_args['train_steps_central']
        self.print_report = train_args['print_report']
        self.path_scalers = train_args['path_scalers']
        self.loss_weights_central = train_args['loss_weights_central']
        self.callbacks_args = train_args['callbacks_args']

        self.save_train_settings(self.model_dir)


    def save_train_settings(self, save_dir):

        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)

        self.metadata = {}
        self.metadata['hidden_size_bands'] = self.size_hidden_bands
        self.metadata['hidden_size_central'] = self.size_hidden_central
        self.metadata['rnn_layers_bands'] = self.rnn_layers_bands
        self.metadata['rnn_layers_central'] = self.rnn_layers_central
        self.metadata['fc_layers_bands'] = self.fc_layers_bands
        self.metadata['fc_layers_central'] = self.fc_layers_central
        self.metadata['regression_size'] = self.regression_size
        self.metadata['buffer_size'] = self.buffer_size
        self.metadata['epochs'] = self.epochs
        self.metadata['num_cores'] = self.num_cores
        self.metadata['batch_size'] = self.batch_size
        self.metadata['dropout'] = self.dropout
        self.metadata['lr_bands'] = self.lr_bands
        self.metadata['lr_central'] = self.lr_central
        self.metadata['val_steps'] = self.val_steps
        self.metadata['max_to_keep'] = self.max_to_keep
        self.metadata['w'] = self.w
        self.metadata['s'] = self.s
        self.metadata['Number of classes'] = self.num_classes
        self.metadata['Number of bands'] = self.n_bands
        self.metadata['Physical_parameters'] = self.physical_params
        self.metadata['mode'] = self.mode

        self.metadata['Max per class'] = self.max_L
        self.metadata['Min per class'] = self.min_L
        self.metadata['Max points per lc'] = self.max_N
        self.metadata['Min points per lc'] = self.min_N

        class_keys = self.trans
        keys = [str(k) for k in class_keys.keys()]
        class_keys = dict(zip(keys, class_keys.values()))
        self.metadata['class_keys'] = class_keys

        path = save_dir+'/metadata_train.json'
        with open(path, 'w') as fp:
            json.dump(self.metadata, fp)

    # def __loss_CrossEntropy(self):
    #     # Create a loss function that adds the MSE loss (or the arguments of the function)
    #     if self.use_class_weights:
    #         @tf.function( experimental_relax_shapes=True)
    #         def loss(y_true,y_pred):
    #             values = tf.keras.losses.categorical_crossentropy(y_true,
    #                                                               y_pred,
    #                                                               from_logits=False,
    #                                                               label_smoothing=0.2,
    #                                                               )
    #             weights = tf.reduce_sum(tf.multiply(tf.cast(y_true, tf.float32), self.vector_weights), axis=1)
    #             values = tf.multiply(weights, values)
    #             values = tf.reduce_mean(values)
    #             return values
    #     else:
    #         @tf.function( experimental_relax_shapes=True)
    #         def loss(y_true,y_pred):
    #             values = tf.keras.losses.categorical_crossentropy(y_true,
    #                                                               y_pred,
    #                                                               from_logits=False,
    #                                                               label_smoothing=0.0,
    #                                                               )
    #             # weights = tf.reduce_sum(tf.multiply(tf.cast(y_true, tf.float32), self.vector_weights), axis=1)
    #             # values = tf.multiply(weights, values)
    #             values = tf.reduce_mean(values)
    #             return values
    #     # Return a function
    #     return loss
    #
    # def loss_CrossEntropy_Full(self, N_skip=5):
    #     @tf.function( experimental_relax_shapes=True)
    #     def loss(y_true,y_pred):
    #
    #         mask = tf.not_equal(y_pred[:,:,0], -1.0)
    #         mask = tf.cast(mask, tf.float32)
    #
    #         # # Get the length of each sequence
    #         N =tf.reduce_sum(mask, axis=1)
    #
    #         # Mask always the first N_skip steps
    #
    #         # All 1 tensor (the ones we want to skip)
    #         m11 = tf.ones(shape=(tf.shape(mask)[0], N_skip), dtype=tf.float32)
    #         # All ones,( the padding) Note the shape
    #         m12 = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1]-N_skip), dtype=tf.float32)
    #         # Concat both tensors along the time dimension
    #         m1 = tf.concat((m11, m12), axis=1)
    #         # Substract the first steps to the real mask
    #         mask = mask-m1
    #
    #         # Find the number of timesteps
    #         reps = tf.shape(mask)[1]
    #         # Repeat the label along the time dimension (1)
    #         y_true = tf.expand_dims(y_true, 1)
    #         y_true = tf.repeat(y_true,[reps],axis=1)
    #
    #         values = tf.keras.losses.categorical_crossentropy(y_true,
    #                                                           y_pred,
    #                                                           from_logits=False,
    #                                                           label_smoothing=0.0,
    #                                                           )
    #         # Multiply by the float32 mask
    #         values = tf.multiply(values, mask)
    #         # mean over the batch
    #         values = tf.reduce_mean(values)
    #         return values
    #     # Return a function
    #     return loss

    # def __loss_MSE():
    #     # Create a loss function that adds the MSE loss (or the arguments of the function)
    #     @tf.function( experimental_relax_shapes=True)
    #     def loss(y_true,y_pred):
    #         #Create mask for y_true
    #         bol = tf.cast(tf.math.not_equal(y_true,0.0), tf.float32) # Values with measurements
    #         val = self.MSE(y_true, y_pred, sample_weight=bol)
    #         return val
    #     # Return a function
    #     return loss

    # @tf.function( experimental_relax_shapes=True)
    # def __last_relevant(self, output, length):
    #     '''Get the last relevant output from the network'''
    #     batch_size = tf.shape(output)[0]
    #     max_length = tf.shape(output)[1]
    #     out_size = int(output.get_shape()[2])
    #     index = tf.range(0, batch_size) * max_length + (length - 1)
    #     flat = tf.reshape(output, [-1, out_size])
    #     relevant = tf.gather(flat, index, name= 'RNN_out')
    #     return relevant

    # def __sort_states(self, tensor_test, indices):
    #     '''Sort tensor_test given the order in indices'''
    #     shapes = tf.shape(tensor_test, name='Get_shapes')
    #     M = shapes[0]
    #     N = shapes[1]
    #     X, Y = tf.meshgrid(tf.range(0,N), tf.range(0,M))
    #     tf_indices = tf.stack([Y,indices], axis=2)
    #     sorted_tensor = tf.gather_nd(tensor_test, tf_indices)
    #     return sorted_tensor

    # def __get_optim(self, lr, optimizer='Adam'):
    #     # Specify the scheduler
    #     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr,
    #                                                                  decay_steps=50,
    #                                                                  decay_rate=0.95,
    #                                                                  staircase=False)
    #     # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr,
    #     #                                                         decay_steps=300,
    #     #                                                         ) in V 2.5.0
    #     # Specify which optimizer yo use
    #     # if optimizer=='AdamW':
    #     #     # weight_schedule =  tf.keras.optimizers.schedules.ExponentialDecay(lr*1e-2,
    #     #     #                                                                  decay_steps=500,
    #     #     #                                                                  decay_rate=0.98,
    #     #     #                                                                  staircase=False)
    #     #     optim = tfa.optimizers.AdamW(learning_rate=lr_schedule,
    #     #                                  weight_decay = 1e-4,
    #     #                                  )
    #     # else:
    #     optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule,
    #                                      beta_1=0.8,
    #                                      **{'clipvalue':0.05})
    #     return optim

    # def __create_RNNs_bands(self,
    #                         hidden_sizes,
    #                         index,
    #                         common_kernel_layer=None,
    #                         common_recurrent_kernel_layer=None,
    #                         use_mod_cell = False,
    #                         implementation=1,
    #                         return_sequences = True,
    #                         return_state = False,
    #                         bidirectional = False,
    #                         use_gated_common=False):
    #     '''Creates RNNs for each band. It can be implemented with the custom GRU implementation
    #     the CUDnn implementations.'''
    #
    #
    #     # Satisfy the conditions to use cudnn kernel
    #     cudnn_kwargs = { 'activation' : 'tanh',
    #                      'recurrent_activation':'sigmoid',
    #                      'use_bias': True,
    #                      'reset_after' : True,
    #                      }
    #     # Add other parameters
    #     cudnn_kwargs['implementation'] = implementation
    #     # cudnn_kwargs['return_sequences'] = return_sequences
    #
    #     if not bidirectional:
    #         rnns = [None]*len(hidden_sizes)
    #
    #         for i in range(len(hidden_sizes)):
    #             if use_mod_cell:
    #                 gru_cell = ModGRUCell(units=hidden_sizes[i],
    #                                      common_kernel_layer=common_kernel_layer[i],
    #                                      common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
    #                                      kernel_regularizer= tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                      bias_regularizer = tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                      name='GRUCell'+str(i)+'_'+str(index),
    #                                      use_gated_common=use_gated_common,
    #                                      **cudnn_kwargs,
    #                                      )
    #                 rnns[i] =  tf.keras.layers.RNN(gru_cell,
    #                                               return_sequences=return_sequences,
    #                                               return_state=return_state,
    #                                               unroll=False,
    #                                               stateful=False,
    #                                               name='RNN'+str(i)+'_'+str(index),
    #                                               )
    #             elif not use_mod_cell:
    #                 cudnn_kwargs['recurrent_dropout'] = 0
    #                 cudnn_kwargs['unroll'] = False
    #                 cudnn_kwargs['stateful'] = False
    #                 cudnn_kwargs['return_sequences'] = return_sequences
    #                 rnns[i] =  tf.keras.layers.GRU(units=hidden_sizes[i],
    #                                                name='RNN'+str(i)+'_'+str(index),
    #                                                **cudnn_kwargs
    #                                                )
    #
    #     elif bidirectional:
    #         rnns = [[],[]]
    #         if use_mod_cell:
    #             rnns = [[],[]]
    #             # Satisfy the conditions to use cudnn kernel
    #             cudnn_kwargs = { 'activation' : 'tanh',
    #                              'recurrent_activation':'sigmoid',
    #                              'use_bias': True,
    #                              'reset_after' : True,
    #                              }
    #             # Add other parameters
    #             cudnn_kwargs['implementation'] = implementation
    #             for i in range(len(hidden_sizes)):
    #                 gru_cell_f = ModGRUCell(  units=hidden_sizes[i],
    #                                          common_kernel_layer=common_kernel_layer[i],
    #                                          common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
    #                                          kernel_regularizer= tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                          bias_regularizer = tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                          name='GRUCell'+str(i)+'_'+str(index),
    #                                          use_gated_common=use_gated_common,
    #                                          **cudnn_kwargs,
    #                                          )
    #
    #                 rnns_f =  tf.keras.layers.RNN( gru_cell_f,
    #                                               return_sequences = cudnn_kwargs['return_sequences'],
    #                                               return_state=return_state,
    #                                               unroll=cudnn_kwargs['unroll'],
    #                                               stateful=cudnn_kwargs['stateful'],
    #                                               name='RNN_f_'+str(i)+'_'+str(index),
    #                                               go_backwards=False
    #                                               )
    #
    #                 gru_cell_b = ModGRUCell(  units=hidden_sizes[i],
    #                                          common_kernel_layer=common_kernel_layer[i],
    #                                          common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
    #                                          kernel_regularizer= tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                          bias_regularizer = tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
    #                                          name='GRUCell'+str(i)+'_'+str(index),
    #                                          use_gated_common=use_gated_common,
    #                                          **cudnn_kwargs,
    #                                          )
    #                 rnns_b =  tf.keras.layers.RNN(gru_cell_b,
    #                                               return_sequences=cudnn_kwargs['return_sequences'],
    #                                               return_state=return_state,
    #                                               unroll=cudnn_kwargs['unroll'],
    #                                               stateful=cudnn_kwargs['stateful'],
    #                                               name='RNN_b_'+str(i)+'_'+str(index),
    #                                               go_backwards=True
    #                                               )
    #                 rnns[0].append(rnns_f)
    #                 rnns[1].append(rnns_b)
    #
    #         elif not use_mod_cell:
    #             for i in range(len(hidden_sizes)):
    #                 rnns_f = tf.keras.layers.GRU(  units=hidden_sizes[i],
    #                                               name='RNN_f_'+str(i)+'_'+str(index),
    #                                               go_backwards=False,
    #                                               **cudnn_kwargs
    #                                               )
    #                 rnns_b = tf.keras.layers.GRU(  units=hidden_sizes[i],
    #                                               name='RNN_b_'+str(i)+'_'+str(index),
    #                                               go_backwards=True,
    #                                               **cudnn_kwargs
    #                                               )
    #                 rnns[0].append(rnns_f)
    #                 rnns[1].append(rnns_b)
    #
    #     return rnns

    # def __create_RNNs_central(  self,
    #                             hidden_sizes,
    #                             implementation=1,
    #                             return_sequences = True,
    #                             return_state = False,
    #                             bidirectional = False):
    #     '''Creates the central RNNs using the CUDnn GRU implementation.'''
    #
    #     # Satisfy the conditions to use cudnn kernel
    #     cudnn_kwargs = {  'activation' : 'tanh',
    #                       'recurrent_activation':'sigmoid',
    #                       'recurrent_dropout':0,
    #                       'unroll' : False,
    #                       'use_bias': True,
    #                       'reset_after' : True,
    #                       'stateful':False}
    #
    #     # Add other parameters
    #     cudnn_kwargs['return_sequences'] = True
    #     cudnn_kwargs['return_state'] = False
    #
    #     if not bidirectional:
    #         rnns = [None]*len(hidden_sizes)
    #         for i in range(len(hidden_sizes)):
    #             gru = tf.keras.layers.GRU(units=hidden_sizes[i],
    #                                       **cudnn_kwargs,
    #                                       name='GRU'+str(i)+'_Central')
    #             rnns[i] = gru
    #
    #     elif bidirectional:
    #         rnns = [[],[]]
    #         # Satisfy the conditions to use cudnn kernel
    #
    #         for i in range(len(hidden_sizes)):
    #             cudnn_kwargs['go_backwards'] = True
    #             gru_b = tf.keras.layers.GRU(units=hidden_sizes[i],
    #                                       **cudnn_kwargs,
    #                                       name='GRU_b'+str(i)+'_Central')
    #             cudnn_kwargs['go_backwards'] = False
    #             gru_f = tf.keras.layers.GRU(units=hidden_sizes[i],
    #                                       **cudnn_kwargs,
    #                                       name='GRU_f'+str(i)+'_Central')
    #             rnns[0].append(gru_f)
    #             rnns[1].append(gru_b)
    #     return rnns

    def __run_RNNs(self, rnns, input_, mask_, backwards = False):
        '''Excecute the rnns based on the input.
        outputs the last states of the last layer.'''
        # I have to propagate masks

        outputs = []
        output = rnns[0](inputs = input_, mask=mask_)
        outputs.append(output)
        for i in range(1, len(rnns)):
            output = rnns[i](inputs = output, mask=mask_)
            outputs.append(output)

        # Flip the output from the backwards RNN if applicable
        if backwards:
            for j in range(len(rnns)):
                outputs[j] = tf.reverse(outputs[i], [1])
        return outputs

    def __add_writers(self):
        train_log_dirs = [None]*self.n_bands
        self.train_summary_writers = [None]*self.n_bands
        val_log_dirs = [None]*self.n_bands
        self.val_summary_writers = [None]*self.n_bands

        for i in range(self.n_bands):
            train_log_dirs[i] = self.log_folder_train + str(i)
        train_log_dir_C = self.log_folder_train + 'Central'

        for i in range(self.n_bands):
            self.train_summary_writers[i] = tf.summary.create_file_writer(train_log_dirs[i], max_queue=5, flush_millis=1000)
        self.train_summary_writer_C = tf.summary.create_file_writer(train_log_dir_C, max_queue=5, flush_millis=1000)

        for i in range(self.n_bands):
            val_log_dirs[i] = self.log_folder_val + str(i)
        val_log_dir_C = self.log_folder_val + 'Central'

        for i in range(self.n_bands):
            self.val_summary_writers[i] = tf.summary.create_file_writer(val_log_dirs[i], max_queue=5, flush_millis=1000)
        self.val_summary_writer_C = tf.summary.create_file_writer(val_log_dir_C, max_queue=5, flush_millis=1000)

    def train(self, train_args, tfrecords_train, tfrecords_val):
        self.set_train_settings(train_args)
        self.initialize_datasets(tfrecords_train, tfrecords_val)
        self.__define_inputs()
        self.__add_placehoders()
        self.__add_writers()
        self.__add_models()
        self.train_loop()

    def __add_placehoders(self):
        # Boolean mask for the RNN
            # self.masked_input = [None]*self.n_bands

        # Create and stacks the cells
            # self.cells = [None]*self.n_bands
            # self.stacked_cells = [None]*self.n_bands

        # Create and run the RNNs
        self.RNNs = [None]*self.n_bands
        self.rnn_outputs = [None]*self.n_bands

        # Define the outputs, in case we want more from the RNN
        # THIS HAS TO CHANGE
        self.rnn_output = [None]*self.n_bands

        # Add the dense layers to proyect to the classes
            # self.denses = [None]*self.n_bands
            # self.proyections = [None]*self.n_bands
        # Get the predictions
        self.predictions_prob = [None]*self.n_bands
            # self.predictions_index = [None]*self.n_bands
            # self.predictions_class = [None]*self.n_bands
            # self.softmaxs = [None]*self.n_bands
        self.models = [None]*self.n_bands
        self.Ns = [None]*self.n_bands
            # self.out_states = [None]*self.n_bands
            # self.out_state = [None]*self.n_bands
        # self.metrics_prediction = [tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall()]
            # self.masks = [None]*self.n_bands

            # self.orders = [None]*self.n_bands
            # self.outs = [None]*self.n_bands

            # self.out_states = [None]*self.n_bands
            # self.out_proyections = [None]*self.n_bands
            # self.orders = [None]*self.n_bands
            # self.out_inputs = [None]*self.n_bands

        self.logs_train = [None]*self.n_bands
        self.logs_val = [None]*self.n_bands
        self.outputs_ = [None]*self.n_bands

        self.common_kernel = [None]*len(self.size_hidden_bands)
        self.common_recurrent_kernel = [None]*len(self.size_hidden_bands)

        self.band_output_kernels = [None]*self.n_bands
        self.central_output_kernel = [None]*len(self.size_hidden_central)

        self.optimizers = {}
        self.loss_functions = {}
        self.train_metrics = {}

        self.mask_value = -99.99
        self.sauce = [None]*self.n_bands
    def __define_inputs_test(self):
        '''Define the inputs for the test.
        We hardcoded the inputs for the training using the train dataset.'''
        # Define the keys from the dataset
        keys = list(self.dataset_test.element_spec[0].keys())
        self.inputs = {}
        self.inputs_central = {}
        for key in keys:
          self.inputs[key] = tf.keras.layers.Input(shape=self.dataset_test.element_spec[0][key].shape[1:],
                                                   dtype=self.dataset_test.element_spec[0][key].dtype,
                                                   name=key
                                                   )
          self.inputs_central[key] = tf.keras.layers.Input(shape=self.dataset_test.element_spec[0][key].shape[1:],
                                                           dtype=self.dataset_test.element_spec[0][key].dtype,
                                                           name=key
                                                           )
    def __define_inputs(self):
        # Define the keys from the dataset
        keys = list(self.dataset_train.element_spec[0].keys())
        self.inputs = {}
        self.inputs_central = {}
        for key in keys:
          self.inputs[key] = tf.keras.layers.Input(shape=self.dataset_train.element_spec[0][key].shape[1:],
                                                   dtype=self.dataset_train.element_spec[0][key].dtype,
                                                   name=key
                                                   )
          self.inputs_central[key] = tf.keras.layers.Input(shape=self.dataset_train.element_spec[0][key].shape[1:],
                                                           dtype=self.dataset_train.element_spec[0][key].dtype,
                                                           name=key
                                                           )

    # def initialize_datasets(self,
    #                           filename_train,
    #                           filename_val):
    #
    #     loader = Parser.Parser(physical_parameters = self.physical_params,
    #                            n_bands = self.n_bands,
    #                            num_classes = self.num_classes,
    #                            w = self.w,
    #                            batch_size = self.batch_size,
    #                            num_threads = self.num_cores,
    #                            buffer_size =self.buffer_size,
    #                            mode = self.mode,
    #                            )
    #     self.dataset_train = loader.get_dataset(  filename= filename_train,
    #                                               epochs= self.epochs,
    #                                               )
    #     self.dataset_val = loader.get_dataset(  filename=filename_val,
    #                                             epochs = 1,
    #                                             )


    def __create_output_layers(self):
        for i in range(self.n_bands):
            self.band_output_kernels[i] = tf.keras.layers.Dense(
                                                self.size_hidden_bands[-1],
                                                activation='tanh',
                                                use_bias=True,
                                                kernel_initializer='glorot_uniform',
                                                bias_initializer='glorot_uniform',
                                                name='Band_Output_Layer_'+str(i)
                                                )



    def __create_common_layers(self, implementation_=1, reset_after=True):

        layer_name_dict={0: 'z', 1:'r', 2:'h'}
        default_params = {'activation':None,
                          'use_bias':True,
                          'kernel_initializer':'glorot_uniform',
                          'bias_initializer':'zeros',
        }
        for b in range(len(self.size_hidden_bands)):
            if implementation_==1:
                default_params['units'] = self.size_hidden_bands[b]
                self.common_kernel[b] = []
                for l in range(3):
                    default_params['name'] = 'Common_kernel_'+layer_name_dict[l]
                    self.common_kernel[b].append(custom_Dense(**default_params))

                default_params['units'] =self.size_hidden_bands[b]
                self.common_recurrent_kernel[b] = []
                for l in range(3):
                    default_params['name']='Common_recurrent_kernel_'+layer_name_dict[l]
                    self.common_recurrent_kernel[b].append(custom_Dense(**default_params) )
            else:
                default_params['units'] =3*self.size_hidden_bands[b]
                default_params['name'] = 'Common_kernel'
                self.common_kernel[b] = custom_Dense(**default_params)

                if reset_after:
                    default_params['units']=3*self.size_hidden_bands[b]
                    default_params['name']='Common_recurrent_kernel'
                    self.common_recurrent_kernel[b] = custom_Dense(**default_params)
                else:
                    default_params['units']=2*self.size_hidden_bands[b]
                    default_params['name']='Common_recurrent_kernel_inner'
                    l1 = custom_Dense(**default_params)

                    default_params['units']=self.size_hidden_bands[b]
                    default_params['name']='Common_recurrent_kernel_outer'
                    l2 = custom_Dense(**default_params)

                    self.common_recurrent_kernel[b] = [l1,l2]

    # def test_loop(self
    #               ,filename_test
    #               ):
    #     '''Run the test loop.'''
    #
    #     # Initialize test dataset
    #     loader = Parser.Parser(physical_parameters = self.physical_params,
    #                            n_bands = self.n_bands,
    #                            num_classes = self.num_classes,
    #                            w = self.w,
    #                            batch_size = self.batch_size,
    #                            num_threads = self.num_cores,
    #                            buffer_size =self.buffer_size,
    #                            mode = self.mode,
    #                            )
    #     self.dataset_test = loader.get_dataset( filename = filename_test,
    #                                             epochs = 1,
    #                                             shuffle=False,
    #                                             )
    #     # Ground truth and ID
    #     ID = []
    #     Class = []
    #     for batch in self.dataset_test:
    #         ID.append(batch[0]['ID'].numpy())
    #         Class.append(batch[1]['Class'].numpy().argmax(axis=1))
    #     ID = np.array([j for i in ID for j in i]).astype(str)
    #     Class = np.array([self.trans[j] for i in Class for j in i])
    #
    #     # Predict
    #     output = {}
    #     output['Probability'] = self.model_central.predict(self.dataset_test)['Class']
    #     output['Prediction'] = np.vectorize(self.trans.get)(output['Probability'].argmax(axis=1))
    #
    #     # Add everything into the output dict
    #     output['ID'] = ID
    #     output['Class'] = Class
    #     return output
