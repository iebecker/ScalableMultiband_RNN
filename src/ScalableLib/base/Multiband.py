import json
import os

import tensorflow as tf
from datetime import datetime
import numpy as np

class Network:
    def __init__(self):
        self.dataset_test = None
        self.dataset_train = None
        tf.keras.backend.clear_session()

    def load_preprocess_data(self, path):
        with open(path) as f:
            metadata = json.load(f)
            self.w = int(metadata['w'])
            # Multiply w by the number of columns included
            self.w = int(2 * self.w)
            self.s = int(metadata['s'])
            self.n_bands = int(metadata['Number of bands'])
            self.max_l = int(metadata['Max per class'])
            self.min_l = int(metadata['Min per class'])
            self.max_N = int(metadata['Max points per lc'])
            self.min_n = int(metadata['Min points per lc'])
            self.num_classes = int(metadata['Number of classes'])

            trans = metadata['Classes Info']['Keys']

            keys = [int(k) for k in trans.keys()]
            self.trans = dict(zip(keys, trans.values()))
            self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))
            # Get numbers per class
            self.element_class = metadata['Classes Info']['Train set']
            self.element_class = {self.trans_inv[i]: int(self.element_class[i]) for i in self.element_class.keys()}

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

        # Whether to use class weights or not
        self.use_class_weights = train_args['use_class_weights']
        if self.use_class_weights:
            total = np.sum([self.element_class[i] for i in self.element_class]) / self.num_classes
            self.class_weights = {i: total / self.element_class[i] for i in self.element_class}
            self.numpy_weights = np.array([self.class_weights[i] for i in range(self.num_classes)])
            self.vector_weights = tf.constant(self.numpy_weights, dtype=tf.float32)
        else:
            self.class_weights = None
            self.numpy_weights = np.array([1.0 for i in range(self.num_classes)])
            self.vector_weights = tf.constant([1.0 for i in range(self.num_classes)], dtype=tf.float32)

        # Use current time to identify models
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        log_folder = os.path.join(train_args['save_dir'], 'Logs', current_time)
        self.log_folder_train = os.path.join(log_folder, 'train')
        self.log_folder_val = os.path.join(log_folder, 'val')

        self.mode = train_args['mode']

        self.model_dir = os.path.join(train_args['save_dir'], 'Models', current_time)

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

        self.metadata['Max per class'] = self.max_l
        self.metadata['Min per class'] = self.min_l
        self.metadata['Max points per lc'] = self.max_N
        self.metadata['Min points per lc'] = self.min_n

        class_keys = self.trans
        keys = [str(k) for k in class_keys.keys()]
        class_keys = dict(zip(keys, class_keys.values()))
        self.metadata['class_keys'] = class_keys

        path = save_dir + '/metadata_train.json'
        with open(path, 'w') as fp:
            json.dump(self.metadata, fp)

    @staticmethod
    def __run_rnns(rnns, input_, mask_, backwards=False):
        """Execute the rnns based on the input.
        outputs the last states of the last layer."""
        # I have to propagate masks

        outputs = []
        output = rnns[0](inputs=input_, mask=mask_)
        outputs.append(output)
        for i in range(1, len(rnns)):
            output = rnns[i](inputs=output, mask=mask_)
            outputs.append(output)

        # Flip the output from the backwards RNN if applicable
        if backwards:
            for j in range(len(rnns)):
                outputs[j] = tf.reverse(outputs[i], [1])
        return outputs

    def __add_writers(self):
        train_log_dirs = [None] * self.n_bands
        self.train_summary_writers = [None] * self.n_bands
        val_log_dirs = [None] * self.n_bands
        self.val_summary_writers = [None] * self.n_bands

        for i in range(self.n_bands):
            train_log_dirs[i] = self.log_folder_train + str(i)
        train_log_dir_c = self.log_folder_train + 'Central'

        for i in range(self.n_bands):
            self.train_summary_writers[i] = tf.summary.create_file_writer(train_log_dirs[i], max_queue=5,
                                                                          flush_millis=1000)
        self.train_summary_writer_C = tf.summary.create_file_writer(train_log_dir_c, max_queue=5, flush_millis=1000)

        for i in range(self.n_bands):
            val_log_dirs[i] = self.log_folder_val + str(i)
        val_log_dir_c = self.log_folder_val + 'Central'

        for i in range(self.n_bands):
            self.val_summary_writers[i] = tf.summary.create_file_writer(val_log_dirs[i], max_queue=5, flush_millis=1000)
        self.val_summary_writer_C = tf.summary.create_file_writer(val_log_dir_c, max_queue=5, flush_millis=1000)

    def train(self, train_args, tfrecords_train, tfrecords_val):
        self.set_train_settings(train_args)
        self.initialize_datasets(tfrecords_train, tfrecords_val)
        self.__define_inputs()
        self.__add_placeholders()
        self.__add_writers()
        self.__add_models()

    def __add_placeholders(self):
        # Create and run the RNNs
        self.RNNs = [None] * self.n_bands
        self.rnn_outputs = [None] * self.n_bands

        # Define the outputs, in case we want more from the RNN
        # THIS HAS TO CHANGE
        self.rnn_output = [None] * self.n_bands
        # Get the predictions
        self.predictions_prob = [None] * self.n_bands
        self.models = [None] * self.n_bands
        self.Ns = [None] * self.n_bands

        self.logs_train = [None] * self.n_bands
        self.logs_val = [None] * self.n_bands
        self.outputs_ = [None] * self.n_bands

        self.common_kernel = [None] * len(self.size_hidden_bands)
        self.common_recurrent_kernel = [None] * len(self.size_hidden_bands)

        self.band_output_kernels = [None] * self.n_bands
        self.central_output_kernel = [None] * len(self.size_hidden_central)

        self.optimizers = {}
        self.loss_functions = {}
        self.train_metrics = {}

        self.mask_value = -99.99
        self.sauce = [None] * self.n_bands

    def __define_inputs_test(self):
        """Define the inputs for the test.
        We hardcoded the inputs for the training using the train dataset."""
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

    def __create_output_layers(self):
        for i in range(self.n_bands):
            self.band_output_kernels[i] = tf.keras.layers.Dense(
                self.size_hidden_bands[-1],
                activation='tanh',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform',
                name='Band_Output_Layer_' + str(i)
            )
