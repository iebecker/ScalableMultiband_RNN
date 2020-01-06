import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import tensorflow as tf
# sys.path.append('/home/chispa/Dropbox/MyPythonClasses/')

class Network():
    def load_preprocess_data(self, path):
        with open(path) as f:
            metadata = json.load(f)
            self.version = metadata['Version']
            self.w = int(metadata['w'])
            self.info = list(metadata['info'])
            # Multiply w by the number of columns included
            self.w = int(np.sum(self.info)*self.w)
            self.s = int(metadata['s'])
            self.n_bands = int(metadata['Number of bands'])
            self.max_L = int(metadata['Max per class'])
            self.min_L = int(metadata['Min per class'])
            self.max_N = int(metadata['Max points per lc'])
            self.min_N = int(metadata['Min points per lc'])
            self.num_classes= int(metadata['Number of classes'])

            self.size_train = int(metadata['Classes Info']['Train set']['Total'])
            self.size_test = int(metadata['Classes Info']['Test set']['Total'])
            self.size_val = int(metadata['Classes Info']['Val set']['Total'])
            trans = metadata['Classes Info']['Keys']
            keys = [int(k) for k in trans.keys()]
            self.trans = dict(zip(keys, trans.values()))

    def set_train_settings(self, metadata_pre_path, size_hidden=None, rnn_layers=None, fc_layers=None, freq_size=None,
        buffer_size=None, epochs=None, num_cores=None, batch_size=None, dropout=None, lr=None, val_steps=None, max_to_keep=None,
        save_dir='./'):

        self.load_preprocess_data(metadata_pre_path)

        self.size_hidden_bands = size_hidden[:-1]
        self.size_hidden_central = size_hidden[-1]

        self.rnn_layers_bands = rnn_layers[:-1]
        self.rnn_layers_central = rnn_layers[-1]

        self.fc_layers_bands = fc_layers[:-1]
        self.fc_layers_central = fc_layers[-1]

        self.freq_layer_size = freq_size

        self.buffer_size = buffer_size
        self.epochs = epochs
        self.num_cores = num_cores
        self.batch_size = batch_size
        self.dropout = dropout
        self.lr = lr
        self.val_steps = val_steps
        self.max_to_keep = max_to_keep

        log_folder = save_dir + 'Logs/'
        self.log_folder_train = log_folder + 'train/'
        self.log_folder_val = log_folder + 'val/'

        plot_folder = save_dir + 'Plots/'
        self.plot_folder_train = plot_folder + 'train/'
        self.plot_folder_val = plot_folder + 'val/'


        self.model_dir = save_dir + 'Model/'

        # Define parser structure
        self.parser_structure()

        self.save_train_settings(self.model_dir)

    def save_train_settings(self, save_dir):

        if not os.path.exists(save_dir):
            print(save_dir)
            os.makedirs(save_dir)

        self.metadata = {}
        self.metadata['size_hidden_bands'] = self.size_hidden_bands
        self.metadata['size_hidden_central'] = self.size_hidden_central
        self.metadata['rnn_layers_bands'] = self.rnn_layers_bands
        self.metadata['rnn_layers_central'] = self.rnn_layers_central
        self.metadata['fc_layers_bands'] = self.fc_layers_bands
        self.metadata['fc_layers_central'] = self.fc_layers_central
        self.metadata['freq_layer_size'] = self.freq_layer_size
        self.metadata['buffer_size'] = self.buffer_size
        self.metadata['epochs'] = self.epochs
        self.metadata['num_cores'] = self.num_cores
        self.metadata['batch_size'] = self.batch_size
        self.metadata['dropout'] = self.dropout
        self.metadata['lr'] = self.lr
        self.metadata['val_steps'] = self.val_steps
        self.metadata['max_to_keep'] = self.max_to_keep
        self.metadata['w'] = self.w
        self.metadata['num_classes'] = self.num_classes
        self.metadata['n_bands'] = self.n_bands
        class_keys = self.trans
        keys = [str(k) for k in class_keys.keys()]
        class_keys = dict(zip(keys, class_keys.values()))
        self.metadata['class_keys'] = class_keys

        path = save_dir+'metadata_train.json'
        with open(path, 'w') as fp:
            json.dump(self.metadata, fp)

    def load_train_settings(self, metadata_train_path):

        with open(metadata_train_path) as f:
            metadata = json.load(f)
            self.size_hidden_bands = metadata['size_hidden_bands']
            self.size_hidden_central = metadata['size_hidden_central']
            self.n_bands = metadata['n_bands']
            self.rnn_layers_bands = metadata['rnn_layers_bands']
            self.rnn_layers_central = metadata['rnn_layers_central']
            self.fc_layers_bands = metadata['fc_layers_bands']
            self.fc_layers_central = metadata['fc_layers_central']
            self.freq_layer_size = metadata['freq_layer_size']
            self.buffer_size = metadata['buffer_size']
            self.epochs = metadata['epochs']
            self.num_cores = metadata['num_cores']
            self.batch_size = metadata['batch_size']
            self.dropout = metadata['dropout']
            self.lr = metadata['lr']
            self.val_steps = metadata['val_steps']
            self.max_to_keep = metadata['max_to_keep']
            self.window = metadata['window']
            self.num_classes = metadata['num_classes']
            self.trans = metadata['class_keys']
            self.info = metadata['info']
            keys = [int(k) for k in self.trans.keys()]
            self.trans = dict(zip(keys, self.trans.values()))

    def __last_relevant(self,output, length):
        '''Get the last relevant output from the network'''
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    def add_FC(self, prev_layer, layer_size, _name_):
        '''Function to create a fully connected layer.'''
        dense = tf.keras.layers.Dense(units=layer_size, name = 'FC_'+_name_)
        hidden_layer = dense(prev_layer)
        # hidden_layer = tf.contrib.layers.fully_connected(
        #     inputs=prev_layer,
        #     num_outputs=layer_size,
        #     name = 'FC_'+_name_)
        dropout = tf.compat.v1.keras.layers.Dropout(  rate=self.dropout
                                                    , name='Dropout_'+_name_
                                                    , noise_shape=None)
        dropout_layer = dropout(hidden_layer, training=self.is_train)
        # dropout_layer = tf.layers.dropout(
        #     inputs=hidden_layer,
        #     rate=self.dropout,
        #     training=self.is_train,
        #     name='Dropout_'+_name_, noise_shape=None
        #     )
        return dropout_layer

    def parser_structure(self):

        self.context_features = {   'Label': tf.io.FixedLenFeature([],dtype=tf.int64),
                                    'Frequency': tf.io.FixedLenFeature([], dtype=tf.float32),
                                    'ID': tf.io.FixedLenFeature([], dtype=tf.string)}
        self.sequence_features = {}
        for i in range(self.n_bands):
            self.context_features['N_'+str(i)] = tf.io.FixedLenFeature([], dtype=tf.int64)
            self.sequence_features['LightCurve_'+str(i)] = tf.io.VarLenFeature(dtype=tf.float32)
            self.sequence_features['Order_'+str(i)] = tf.io.VarLenFeature(dtype=tf.int64)

    def data_parser(self, serialized_example):
        '''Parse the serialized objects.'''

        context_data, sequence_data = tf.io.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
            )


        # Extract context fe    atures
        # Extract ID
        ID = tf.cast(context_data['ID'], tf.string)

        # Label of the light curve
        Label = tf.cast(context_data['Label'], tf.int32)
        # Encode the label as a hone-hot tensor
        Label = tf.one_hot(Label, self.num_classes, on_value=1, off_value=0, axis=-1, dtype=tf.int32)

        # Extract frequency
        Freq = tf.cast(context_data['Frequency'], tf.float32)

        # Extract info per band
        Ns = [[]]*self.n_bands
        LCs = [[]]*self.n_bands
        Os = [[]]*self.n_bands

        for i in range(self.n_bands):
            # Number of observations per band
            Ns[i] = tf.cast(context_data['N_'+str(i)], tf.int32)
            # Extract light curve representation
            LCs[i] = sequence_data['LightCurve_'+str(i)]
            LCs[i] = tf.compat.v1.sparse.to_dense(LCs[i])
            LCs[i] = tf.cast(LCs[i], tf.float32)
            # Reshape the curves to its original matrix form
            LCs[i] = tf.compat.v1.reshape(LCs[i], [Ns[i], self.w])
            # Extract Order sequence
            Os[i] = sequence_data['Order_'+str(i)]
            Os[i] = tf.compat.v1.sparse.to_dense(Os[i])
            Os[i] = tf.cast(Os[i], tf.int32)
            # Reshape the order to its original form
            Os[i] = tf.compat.v1.reshape(Os[i], [Ns[i]])

        # Output
        output = [0]*(4*self.n_bands+1)
        output[0] = ID
        output[1] = Label
        output[2] = Freq
        for i in range(self.n_bands):
            output[3+i] = LCs[i]
            output[5+i] = Os[i]
            output[7+i] = Ns[i]

        return tuple(output)

    def add_input_iterators(self):
        '''Define the input pipeline graph.'''
        #Place all the excecution in the CPU explicitly.
        with tf.device('/cpu:0'), tf.name_scope('Iterators'):
            self.filename_pl = tf.compat.v1.placeholder(tf.string, shape=[None],name='Filename')
            self.epochs_pl = tf.compat.v1.placeholder(tf.int64, shape=[],name='Epochs')
            self.handle_pl = tf.compat.v1.placeholder(tf.string, shape=[],name='Handle')

            dataset = tf.data.TFRecordDataset(self.filename_pl, num_parallel_reads=3)
            # Repeat epochs_pl times
            dataset = dataset.repeat(count=self.epochs_pl)
            # Deserialize and Parse
            dataset = dataset.map(self.data_parser, num_parallel_calls=self.num_cores)

            self.dataset_train = dataset.shuffle(buffer_size=self.buffer_size)

            # Shapes of the padding Objects
            padd_shapes = [[]]*(4*self.n_bands+1)
            padd_shapes[0] = []
            padd_shapes[1] = [self.num_classes]
            padd_shapes[2] = []
            for i in range(self.n_bands):
                padd_shapes[3+i] = [None, self.w]
                padd_shapes[5+i] = [None]
                padd_shapes[7+i] = []
            padd_shapes = tuple(padd_shapes)
            self.dataset_train = self.dataset_train.padded_batch(self.batch_size, padded_shapes=padd_shapes
                                            ,drop_remainder=False).prefetch(1)
            self.dataset_eval = dataset.padded_batch(self.batch_size, padded_shapes=padd_shapes
                                            ,drop_remainder=False).prefetch(1)

            output_types = tf.compat.v1.data.get_output_types(self.dataset_train)
            output_shapes = tf.compat.v1.data.get_output_shapes(self.dataset_train)
            self.train_iterator = tf.compat.v1.data.Iterator.from_structure(output_types, output_shapes)
            self.train_initializer = self.train_iterator.make_initializer(self.dataset_train)

            self.eval_iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset_eval)
            self.eval_initializer = self.eval_iterator.initializer


            feedable_iter = tf.compat.v1.data.Iterator.from_string_handle(self.handle_pl, output_types, output_shapes)
            self.next_element = feedable_iter.get_next()

    def add_input_placeholders(self):

        self.target_pl = tf.compat.v1.placeholder(tf.float32, [None, self.num_classes],name='Label')
        self.id_pl = tf.compat.v1.placeholder(tf.string, [None],name='ID')
        self.frequency_pl = tf.compat.v1.placeholder(tf.float32, [None], name='Frequency')
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='IsTrain')
        self.global_step = tf.compat.v1.train.get_or_create_global_step()

        self.data_pl = [[]]*self.n_bands
        self.length_pl = [[]]*self.n_bands
        self.order_pl = [[]]*self.n_bands
        for i in range(self.n_bands):
            self.data_pl[i] = tf.compat.v1.placeholder(tf.float32, [None, None, self.w],name='Data_'+str(i))
            self.length_pl[i] = tf.compat.v1.placeholder(tf.int32, [None],name='Length_'+str(i))
            self.order_pl[i] = tf.compat.v1.placeholder(tf.float32, [None, None], name='Order_'+str(i))
        self.data_pl = tuple(self.data_pl)
        self.length_pl  = tuple(self.length_pl )
        self.order_pl = tuple(self.order_pl)
    def __create_fc_layer(self,prev_layer,i):
            '''Function to create a fully connected layer.'''
            hidden_layer = tf.contrib.layers.fully_connected(inputs=prev_layer, num_outputs=self.size_fc
                                                            , normalizer_fn=tf.contrib.layers.batch_norm)
            dropout_layer = tf.layers.dropout(inputs=hidden_layer, rate=self.layer_dropout, training=self.is_train
                                              , name='Dropout_'+str(i), noise_shape=None)
            return dropout_layer

    def GRU_cell(self, size_hidden, _name_):
        const= tf.keras.initializers.Constant(0.1,dtype=tf.float32)
        xavier = tf.contrib.layers.xavier_initializer()
        glorot = tf.glorot_normal_initializer()
        gru_cell = tf.compat.v1.nn.rnn_cell.GRUCell(
            num_units=size_hidden,
            activation=tf.nn.tanh,
            kernel_initializer=glorot,
            bias_initializer=xavier,
            name =_name_
            )

        return gru_cell

    def __func(tensor, t):
        var = tf.broadcast_to(tensor,[t])
        return var

    def add_model(self):
        '''Creates the network to perform classification.'''
        with tf.compat.v1.variable_scope('RNNs'):
            grus=[[]]*self.n_bands
            stacked = [[]]*self.n_bands
            outputs = [[]]*self.n_bands
            states = [[]]*self.n_bands
            lasts = [[]]*self.n_bands
            for i in range(self.n_bands):
                with tf.compat.v1.variable_scope('RNN_'+str(i), reuse=False):
                    grus[i] = [self.GRU_cell(self.size_hidden_bands[i], 'RNN_band_'+str(i)+str(j)) for j in range(self.rnn_layers_bands[i])]
                    stacked[i] = tf.nn.rnn_cell.MultiRNNCell(grus[i],state_is_tuple=True)
                    outputs[i], states[i] = tf.compat.v1.nn.dynamic_rnn(stacked[i], self.data_pl[i], dtype=tf.float32
                                                                , sequence_length=self.length_pl[i]
                                                                , swap_memory=False)
                    lasts[i] = self.__last_relevant(outputs[i], self.length_pl[i])

            with tf.compat.v1.variable_scope('RNNCentral', reuse=False):
                self.Length_Central = 0
                for i in range(self.n_bands):
                    self.Length_Central += self.length_pl[i]

                self.concat_states = tf.concat(outputs,axis=1,name='Concat_states')
                self.concat_orders = tf.concat(self.order_pl, axis=1,name='Concat_order')

                k = tf.shape(self.concat_orders)[1]
                indices =  tf.argsort(self.concat_orders,axis=1,direction='ASCENDING',name='Sort_order')
                shape_orders = tf.shape(self.concat_orders)
                auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(shape_orders[:(self.concat_orders.get_shape().ndims - 1)]) + [k])], indexing='ij')

                self.Data_Central = tf.gather_nd(self.concat_states, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))

                central_grus = [self.GRU_cell(self.size_hidden_central, 'RNN_Central_'+str(j)) for j in range(self.rnn_layers_central)]
                cell_Central =tf.nn.rnn_cell.MultiRNNCell(central_grus, state_is_tuple=True)

                self.output_central, self.state_central = tf.compat.v1.nn.dynamic_rnn(cell_Central, self.Data_Central
                        , dtype=tf.float32, sequence_length=self.Length_Central, swap_memory=False)
                last_Central = self.__last_relevant(self.output_central, self.Length_Central)


        #Add fully connected layers per band
        with tf.compat.v1.variable_scope('Hidden_Layers_Bands'):
            FC_bands = [[None]*len(i) for i in self.fc_layers_bands]
            for i in range(self.n_bands):
                if len(self.fc_layers_bands[i])>0:
                    for j in range(len(self.fc_layers_bands[i])):
                        if j==0:
                            FC_bands[i][j] = self.add_FC(lasts[i], self.fc_layers_bands[i][j], 'band_'+str(j))
                        else:
                            FC_bands[i][j] = self.add_FC(FC_bands[i][j-1], self.fc_layers_bands[i][j], 'band_'+str(j))
                else:
                    FC_bands[i][0] = self.last_h # Fix, I have multiple RNN chains, what happens there?!!!!!!!!!!!!!!!!!!!

        with tf.compat.v1.variable_scope('Hidden_Layers_Central'):
            FC_central = self.fc_layers_central.copy()

            if len(self.fc_layers_central)>0:
                for j in range(len(self.fc_layers_central)):
                    if j==0:
                        FC_central[j] = self.add_FC(last_Central, self.fc_layers_central[j], 'Central_'+str(j))
                    else:
                        FC_central[j] = self.add_FC(FC_central[j-1], self.fc_layers_central[j], 'Central_'+str(j))
            else:
                FC_central = last_Central

        with tf.compat.v1.variable_scope('Frequency_Branch'):
            # Use the attention output.
            self.attention_central = last_Central
            # Creates the boolean mask of the objects without freq
            self.mask = tf.greater(self.frequency_pl, 0, name='Mask')
            self.mask = tf.cast(self.mask, tf.float32)
            # Applies the mask to the data
            self.attention_central = tf.math.multiply(self.mask, self.attention_central , name='Boolean_Mask')


            freq_layers = [[]]*(len(self.freq_layer_size)+1)
            for i in range(len(freq_layers)-1):
                if i>0:
                    in_freq = freq_layers[i-1]
                else:
                    in_freq = self.attention_central
                freq_layers[i] = tf.compat.v1.layers.dense(inputs = in_freq
                                                            , units = self.freq_layer_size[i]
                                                            , activation=tf.compat.v1.sigmoid
                                                            , use_bias=True
                                                            , kernel_initializer= tf.compat.v1.initializers.glorot_uniform
                                                            , bias_initializer =  tf.compat.v1.initializers.glorot_uniform
                                                            , name = 'FC_Freq_'+str(i))
            freq_layers[-1] = tf.compat.v1.layers.dense(inputs = freq_layers[-2]
                                                        , units = 1
                                                        , activation=tf.compat.v1.sigmoid
                                                        , use_bias=True
                                                        , kernel_initializer= tf.compat.v1.initializers.glorot_uniform
                                                        , bias_initializer =  tf.compat.v1.initializers.glorot_uniform
                                                        , name= 'FC_Freq_'+str(i+1))
#
        with tf.compat.v1.variable_scope('Logits'):
            logits = [[]]*(self.n_bands)
            for i in range(self.n_bands):
                logits[i] =tf.compat.v1.layers.dense(inputs = FC_bands[i][-1]
                                                            , units = self.num_classes
                                                            , activation=tf.compat.v1.sigmoid
                                                            , use_bias=True
                                                            , kernel_initializer=tf.compat.v1.initializers.glorot_uniform
                                                            , bias_initializer = tf.compat.v1.initializers.glorot_uniform
                                                            , name = 'Logits_bands_'+str(i))


            # This can and must be modified
            input_logit_central = tf.concat([FC_bands[i][-1] for i in range(self.n_bands)]+[FC_central[-1]], axis=1, name='Concat_Mod')
            logits_central =tf.compat.v1.layers.dense(inputs = input_logit_central
                                            , units = self.num_classes
                                            , activation= tf.compat.v1.sigmoid
                                            , use_bias= True
                                            , kernel_initializer= tf.compat.v1.initializers.glorot_uniform
                                            , bias_initializer = tf.compat.v1.initializers.glorot_uniform
                                            , name = 'Logits_Central')

        with tf.compat.v1.variable_scope('Softmax'):
            predictions = [[]]*self.n_bands
            for i in range(self.n_bands):
                predictions[i] = tf.compat.v1.nn.softmax(logits = logits[i]
                                                        , name = 'Softmax_band_'+str(i))

            prediction_central = tf.compat.v1.nn.softmax(logits = logits_central
                                                    , name = 'Softmax_Central')

        with tf.compat.v1.variable_scope('Index_Predictions'):
            self.index_predictions = [[]]*self.n_bands
            for i in range(self.    n_bands):
                self.index_predictions[i] = tf.argmax(predictions[i], axis=1)

            self.index_prediction_Central = tf.argmax(prediction_central, axis=1)
            self.index_target = tf.argmax(self.target_pl, axis=1)

        with tf.compat.v1.variable_scope('Losses'):
            self.losses = [[]]*self.n_bands
            for i in range(self.n_bands):
                self.losses[i] = tf.keras.backend.categorical_crossentropy(target = self.target_pl
                                                        , output = logits[i]
                                                        , from_logits = True
                                                        )
                self.losses[i] = tf.reduce_mean(self.losses[i], name = 'ReduceMean_band_'+str(i))

            self.loss_central = tf.keras.backend.categorical_crossentropy(target = self.target_pl
                                                    , output = logits_central
                                                    , from_logits = True
                                                    )
            self.loss_central = tf.reduce_mean(self.loss_central, name = 'ReduceMean_Central')


            self.loss_freq = tf.keras.losses.MSE(self.frequency_pl, freq_layers[-1])
            # Mask the loss with 0 if the objects do not have measured period, 1 otherwise.
            self.loss_freq = tf.compat.v1.multiply(self.loss_freq, self.mask)

            self.loss_multitask = tf.add(self.loss_central, self.loss_freq, name = 'Loss_Multitask')
            for i in range(self.n_bands):
                self.loss_multitask = tf.add(self.loss_multitask, self.losses[i], name = 'Loss_Multitask')

        # Transform mistakes to float32 mistakes
        # The reduce mean average the values by batch size
        with tf.compat.v1.variable_scope('Errors'):
            mistakes = [[]]*self.n_bands
            for i in range(self.n_bands):
                mistakes[i] = tf.not_equal(tf.argmax(self.target_pl, 1), tf.argmax(predictions[i], 1), name='Mistakes_band_'+str(i))
            mistakes_Central = tf.not_equal(tf.argmax(self.target_pl, 1), tf.argmax(prediction_central, 1), name='Mistakes_Central')

            self.errors = [[]]*(self.n_bands+2)
            for i in range(self.n_bands):
                self.errors[i] = tf.reduce_sum(tf.cast(mistakes[i], tf.float32), name = 'Error_band_'+str(i))
            self.errors[-2] = tf.reduce_sum(tf.cast(mistakes_Central, tf.float32), name = 'Error_Central')

            self.errors[-1] = tf.reduce_sum(self.loss_freq, name = 'Error_Freq')

        with tf.compat.v1.variable_scope('Optimizers'):
            optimizers = [[]]*self.n_bands
            lrs = [[]]*self.n_bands
            for i in range(self.n_bands):
                lrs[i] =  tf.compat.v1.train.exponential_decay(self.lr, self.global_step, 1000, 0.98, staircase=True)
                optimizers[i] = tf.compat.v1.train.AdamOptimizer(lrs[i], name = 'Optimizer_band_'+str(i))

            lr_central = tf.compat.v1.train.exponential_decay(self.lr, self.global_step, 1000, 0.98, staircase=True)
            lr_freq = tf.compat.v1.train.exponential_decay(self.lr, self.global_step, 1000, 0.98, staircase=True)
            optimizer_Central = tf.compat.v1.train.AdamOptimizer(lr_central, name='Optimizer_Central')
            optimizer_MultiTask = tf.compat.v1.train.AdamOptimizer(lr_freq, name='Optimizer_MultiTask')

        with tf.compat.v1.variable_scope('Trainable_Variables'):
            # Define the specific variables for each band and the central
            trainable_variables = [[]]*(self.n_bands)
            variable_set = set(tf.compat.v1.trainable_variables()) # All the variables in a set
            for i in range(self.n_bands):
                trainable_variables[i] = [j for j in tf.compat.v1.trainable_variables() if '_band_'+str(i) in j.name]
                variable_set.difference_update(set(trainable_variables[i]))

            trainable_variables_central = list(variable_set)

        with tf.compat.v1.variable_scope('Train_Steps'):
            self.train_steps = [[]]*self.n_bands
            for i in range(self.n_bands):
                self.train_steps[i] =  optimizers[i].minimize(self.losses[i]
                                                            , global_step= self.global_step
                                                            ,var_list=trainable_variables[i])
            # self.train_step_Central = self.optimizer_Central.minimize(self.loss_Central, global_step= self.global_step, var_list=self.trainable_variables[-1])
            self.train_step_Multitask = optimizer_Central.minimize(self.loss_multitask
                                                                    , global_step=self.global_step
                                                                    , var_list=trainable_variables_central)

    def add_writers(self):
        graph = tf.compat.v1.get_default_graph()
        self.writer_train = tf.compat.v1.summary.FileWriter(self.log_folder_train, graph, flush_secs=30)
        self.writer_val = tf.compat.v1.summary.FileWriter(self.log_folder_val, graph, flush_secs=30)

    def train(self, train_args, tfrecords_train, tfrecords_val):

        tf.compat.v1.reset_default_graph()
        self.set_train_settings(**train_args)
        self.build_graph()
        self.add_writers()

        with tf.compat.v1.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.local_variables_initializer())

            # Initialize the training iterator
            train_handle = sess.run(self.train_iterator.string_handle())
            iterator_dict = {self.filename_pl: tfrecords_train, self.epochs_pl: self.epochs
                            , self.handle_pl: train_handle}
            sess.run(self.train_initializer, iterator_dict)

            while True:
                try:
                    step = tf.compat.v1.train.global_step(sess, self.global_step)

                    # data, labels, lengths, ids = sess.run(self.next_element, feed_dict={self.handle_pl:train_handle})
                    batch_elements = sess.run(self.next_element, feed_dict={self.handle_pl:train_handle})
                    feed_dict = {self.id_pl: batch_elements[0]
                                , self.target_pl: batch_elements[1]
                                , self.frequency_pl: batch_elements[2]
                                , self.data_pl: batch_elements[3:3+self.n_bands]
                                , self.order_pl: batch_elements[3+self.n_bands:3+2*self.n_bands]
                                , self.length_pl: batch_elements[3+2*self.n_bands:3+3*self.n_bands]
                                , self.is_train: True
                                }


                    # Run train steps per band
                    for i in range(self.n_bands):
                        sess.run(self.train_steps[i], feed_dict)
                    sess.run(self.train_step_Multitask, feed_dict)

                    if step%self.val_steps==0:
                        args_train = [sess, tfrecords_train, self.writer_train, self.plot_folder_train
                        , step, 'Train']
                        args_val = [sess, tfrecords_val, self.writer_val, self.plot_folder_val
                        , step, 'Val']
                        loss_train = self.save_metrics(*args_train)
                        loss_val = self.save_metrics(*args_val)
                        save_path = self.model_dir + 'model.ckpt'
                        self.saver.save(sess, save_path, global_step=step)

                except tf.errors.OutOfRangeError:
                    print('Training ended')
                    self.writer_val.close()
                    self.writer_train.close()
                    break

    def save_metrics(self, sess, tfrecords, writer, plot_folder, step, name):
        # Initialize eval iterator
        handle = sess.run(self.eval_iterator.string_handle())
        iterator_dict = {self.filename_pl: tfrecords, self.epochs_pl: 1, self.handle_pl: handle}
        sess.run(self.eval_initializer, iterator_dict)

        _preds_bands= [np.zeros(0)]*self.n_bands
        _preds_central = np.zeros(0)
        _labels= np.zeros(0)
        _ids = np.zeros(0,dtype=np.int64)
        _freqs = np.zeros(0)
        Loss_bands = [0]*self.n_bands
        Loss_multitask = 0
        tensors = [self.losses, self.loss_multitask, self.index_predictions, self.index_prediction_Central, self.index_target, self.frequency_pl]
        while True:
            try:
                batch_elements = sess.run(self.next_element, feed_dict={self.handle_pl:handle})

                feed_dict = {self.id_pl: batch_elements[0]
                            , self.target_pl: batch_elements[1]
                            , self.frequency_pl: batch_elements[2]
                            , self.data_pl: batch_elements[3:3+self.n_bands]
                            , self.order_pl: batch_elements[3+self.n_bands:3+2*self.n_bands]
                            , self.length_pl: batch_elements[3+2*self.n_bands:3+3*self.n_bands]
                            , self.is_train: True
                            }

                losses, loss_multitask, predictions, prediction_central, target, freqs = sess.run(tensors, feed_dict)

                ### Acumulate results
                # quantity per batch
                batch = batch_elements[0].shape[0]
                # Loss per band
                Loss_bands = [[]]*self.n_bands
                for i in range(self.n_bands):
                     Loss_bands+=losses[i]*batch
                # Loss in the Multitask scheme
                Loss_multitask += loss_multitask*batch

                # Predictions per band
                for i in range(self.n_bands):
                    _preds_bands[i] = np.append(_preds_bands[i], predictions[i])
                # Predictions in the central RNN
                _preds_central = np.append(_preds_central, prediction_central)

                # Frequencies per object
                _freqs = np.append(_freqs, freqs)
                # Class per object
                _labels= np.append(_labels, target)
                # IDs per object
                _ids = np.append(_ids, batch_elements[-2])

            except tf.errors.OutOfRangeError:
                errors = [0]*self.n_bands
                for i in range(self.n_bands):
                    bol = _preds_bands[i]!=_labels
                    errors[i] = np.sum(bol)/bol.shape[0]
                bol = _preds_central!=_labels
                error_central = np.sum(bol)/bol.shape[0]

                print('Prediction accuracy on {} set: {:3.2f}%'.format(name, 100 * (1-error_central)))
                # Add the summaries
                # summary = sess.run(self.summary_op_mod, feed_dict={self.err_sum_ph: err, self.loss_sum_ph: Loss/bol.shape[0]})
                # writer.add_summary(summary, step)
                # writer.flush()

                # self.plot_cm(_labels, _preds, plot_folder, step)
                return Loss_multitask/bol.shape[0]

    def build_graph(self):
        tf.compat.v1.reset_default_graph()
        self.add_input_placeholders()
        self.add_input_iterators()
        self.add_model()
        self.add_saver()

    def add_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.max_to_keep, pad_step_number=True)


    def save_results(self, label_, pred_):
        '''Print the classification results.'''
        label = [self.trans[i] for i in label_]
        pred = [self.trans[i] for i in pred_]
        np.savetxt(self.clasification_save+'/Results_num.dat',np.column_stack([label_, pred_]), fmt='%.1d', header='Target, Result', comments='', delimiter=',')
        np.savetxt(self.clasification_save+'/Results.dat',np.column_stack([label, pred]), fmt='%s', header='Target,Result', comments='', delimiter=',')


    def create_folder(self):
        '''Function to create filders for diferent runs.'''
        self.folders = glob(self.logs_save+'/*')
        if self.folders == []:
            self.nrun = 1
        else:
            var = [int(i.split('_')[-1]) for i in self.folders]
            self.nrun = max(var)+1

    def save_metadata(self):
        '''Save the metadata for each model.'''

        filename = self.log_folder+'/metadata'
        with open(filename, 'w') as f:
            f.write('Epochs ={}\n'.format(self.epochs))
            # f.write('Includes time ={}\n'.format(self.w_time))
            f.write('w ={}\n'.format(self.window))
            f.write('ws ={}\n'.format(self.ws))
            # f.write('quantity cap ={}\n'.format(self.max_L))
            f.write('Batch size ={}\n'.format(self.batch_size))
            f.write('Num recurrent layers ={}\n'.format(self.num_layers))
            f.write('Size recurrent layer ={}\n'.format(self.size_hidden))
            f.write('Learning rate ={}\n'.format(self.learning_rate))
            f.write('Num fully connected layers={}\n'.format(self.n_fc))
            f.write('Size fully connected layer ={}\n'.format(self.size_fc))
            f.write('Cell dropout ={}\n'.format(self.cell_dropout))
            f.write('Layer dropout ={}\n'.format(self.layer_dropout))
            f.write('Num Cores ={}\n'.format(self.num_cores_map))
            f.write('Buffer size ={}\n'.format(self.buffer_size))
            # f.write('Cache input ={}\n'.format(self.cache))

    def plot_confusion_matrix(self, target, prediction,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Greens,
                             save=True, nep=0):

        """ This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        plt.clf()
        cm = confusion_matrix(target, prediction)
        SMALL_SIZE = 15
        MEDIUM_SIZE = 20
        BIGGER_SIZE = 22

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        plt.title(title)
        tick_marks = np.arange(self.num_classes)
        plt.xticks(tick_marks, [self.trans[i] for i in range(self.num_classes)], rotation=45)
        plt.yticks(tick_marks, [self.trans[i] for i in range(self.num_classes)])

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    cm[i,j] ="%.2f" %cm[i,j]

        thresh = 0.001
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment="center"
                     ,color="white" if cm[i, j] < thresh else "black")

        if not save:

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        else:
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.colorbar()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        plt.savefig(self.cms_save+str(nep)+'.png', format='png', dpi=250)
