import tensorflow as tf
from glob import glob

class Parser():
    def __init__(self, n_bands, num_classes, w, batch_size, mode, num_threads,physical_parameters=[], physical_parameters_est=[], buffer_size=10000):
        self.n_bands = n_bands
        self.num_classes = num_classes
        self.w = w
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.buffer_size = buffer_size
        self.phys_params = physical_parameters
        self.phys_params_est = physical_parameters_est
        self.mode = mode
        self.parser_structure()

    # @tf.function
    def parser_structure(self):
        self.context_features = {'Label': tf.io.FixedLenFeature([],dtype=tf.int64),
                                 'ID': tf.io.FixedLenFeature([], dtype=tf.string)}
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.context_features[param] = tf.io.FixedLenFeature([], dtype=tf.float32)
            for param in self.phys_params_est:
                self.context_features[param] = tf.io.FixedLenFeature([], dtype=tf.float32)

        self.sequence_features = {}
        for i in range(self.n_bands):
            self.context_features['N_'+str(i)] = tf.io.FixedLenFeature([], dtype=tf.int64) # Lenth of lc per band
            self.context_features['M0_'+str(i)] = tf.io.FixedLenFeature([], dtype=tf.float32) # First magnitude
            self.context_features['T0_'+str(i)] = tf.io.FixedLenFeature([], dtype=tf.float32) # First time
            self.sequence_features['LightCurve_'+str(i)] = tf.io.VarLenFeature(dtype=tf.float32) # Matrix representation
            self.sequence_features['Order_'+str(i)] = tf.io.VarLenFeature(dtype=tf.int64) # Order of observations
            self.sequence_features['Uncertainty_'+str(i)] = tf.io.VarLenFeature(dtype=tf.float32) # Uncertainty representation

    @tf.function
    def log10(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def get_padd_shapes_and_values(self):
        # Values of padding
        self.padd_values_X={}
        self.padd_values_X['ID'] = tf.constant('', dtype=tf.string)

        # Shapes of the padding Objects
        self.padd_shapes_X = {}
        self.padd_shapes_X['ID'] = []

        for i in range(self.n_bands):
            self.padd_shapes_X['input_LC_'+str(i)] = [None, self.w]
            self.padd_values_X['input_LC_'+str(i)] = tf.constant(0, dtype=tf.float32)

            self.padd_shapes_X['O_'+str(i)] = [None]
            self.padd_values_X['O_'+str(i)] = tf.constant(999, dtype=tf.int32)

            self.padd_shapes_X['N_'+str(i)] = []
            self.padd_values_X['N_'+str(i)] = tf.constant(0, dtype=tf.int32)

            self.padd_shapes_X['M0_'+str(i)] = []
            self.padd_values_X['M0_'+str(i)] = tf.constant(0, dtype=tf.float32)

            self.padd_shapes_X['T0_'+str(i)] = []
            self.padd_values_X['T0_'+str(i)] = tf.constant(0, dtype=tf.float32)

            self.padd_shapes_X['U_'+str(i)] = [None]
            self.padd_values_X['U_'+str(i)] = tf.constant(0, dtype=tf.float32)

        # For the target dictionary
        self.padd_values_Y = {}
        self.padd_values_Y['Class']= tf.constant(0,dtype=tf.int32)
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.padd_values_Y[param] = tf.constant(0.0, dtype=tf.float32)

        # For the light curves
        if 'language' in self.mode:
            for i in range(self.n_bands):
                self.padd_values_Y['LC_'+str(i)]= tf.constant(0, dtype=tf.float32)

        # The shapes of the classifier and regressions
        self.padd_shapes_Y = {}
        self.padd_shapes_Y['Class'] = [self.num_classes,]
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.padd_shapes_Y[param] = []
        if 'language' in self.mode:
            for i in range(self.n_bands):
                self.padd_shapes_Y['LC_'+str(i)] = [None, self.w]


    def data_parser(self, serialized_example):
        '''Parse the serialized objects.'''

        context_data, sequence_data = tf.io.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
            )

        # Extract context features
        # Extract ID
        ID = tf.cast(context_data['ID'], tf.string)
        # Label of the light curve
        Label = tf.cast(context_data['Label'], tf.int32)
        # Encode the label as a hone-hot tensor
        Label = tf.one_hot(Label, self.num_classes, on_value=1, off_value=0, axis=-1, dtype=tf.int32)

        # Extract regression parameters
        if 'regression' in self.mode:
            phys_params = {}
            for param in self.phys_params:
                var = tf.cast(context_data[param], tf.float32)
                phys_params[param] = var
            # Extract regression estimate parameters
            phys_params_est = {}
            for param in self.phys_params_est:
                var = tf.cast(context_data[param], tf.float32)
                phys_params_est[param] = var


        # Extract info per band
        Ms  = [None]*self.n_bands
        Ts  = [None]*self.n_bands
        Ns  = [None]*self.n_bands
        LCs = [None]*self.n_bands
        Os  = [None]*self.n_bands
        Us  = [None]*self.n_bands

        for i in range(self.n_bands):
            # First observation per band
            Ms[i] = tf.cast(context_data['M0_'+str(i)], tf.float32)
            # First time per band
            Ts[i] = tf.cast(context_data['T0_'+str(i)], tf.float32)
            # Number of observations per band
            Ns[i] = tf.cast(context_data['N_'+str(i)], tf.int32)
            # Extract light curve representation
            LCs[i] = sequence_data['LightCurve_'+str(i)]
            LCs[i] = tf.sparse.to_dense(LCs[i])
            LCs[i] = tf.cast(LCs[i], tf.float32)
            # Reshape the curves to its original matrix form
            LCs[i] = tf.reshape(LCs[i], [Ns[i], self.w])
            # Extract Order sequence
            Os[i] = sequence_data['Order_'+str(i)]
            Os[i] = tf.sparse.to_dense(Os[i])
            Os[i] = tf.cast(Os[i], tf.int32)
            # Reshape the order to its original form
            Os[i] = tf.reshape(Os[i], [Ns[i]])
            # Uncertainty containers
            Us[i] = sequence_data['Uncertainty_'+str(i)]
            Us[i] = tf.sparse.to_dense(Us[i])
            Us[i] = tf.cast(Us[i], tf.float32)
            Us[i] = tf.reshape(Us[i], [Ns[i]])


        # output dictionary
        dict_in = {}
        dict_in['ID'] = ID
        for i in range(self.n_bands):
            dict_in['input_LC_'+str(i)] = LCs[i]
            dict_in['O_'+str(i)] = Os[i]
            dict_in['N_'+str(i)] = Ns[i]
            dict_in['M0_'+str(i)] = Ms[i]
            dict_in['T0_'+str(i)] = Ts[i]
            dict_in['U_'+str(i)] = Us[i]


        dict_out = {}
        dict_out['Class'] = Label
        if 'regression' in self.mode:
            for param in self.phys_params:
                dict_out[param] = phys_params[param]
        if 'language' in self.mode:
            for i in range(self.n_bands):
                dict_out['LC_'+str(i)] = LCs[i]

        return (dict_in, dict_out)

    def get_dataset(self,
                    filename,
                    epochs,
                    shuffle = True):
        '''
        Parse the files in filename
        '''
        # If filename contains a wildcard, get all the files
        if '*' in filename:
            filename = glob(filename)

        dataset = tf.data.TFRecordDataset(filename,
                                          num_parallel_reads = tf.data.experimental.AUTOTUNE,
                                          )


        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        # Deserialize and Parse
        dataset = dataset.map(self.data_parser,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE
                              )

        # Repeat epochs times
        dataset = dataset.repeat(count=epochs)



        self.get_padd_shapes_and_values()

        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(self.padd_shapes_X, self.padd_shapes_Y),
                                       drop_remainder=False,
                                       padding_values=(self.padd_values_X, self.padd_values_Y),
                                       )
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.output_types = tf.compat.v1.data.get_output_types(dataset)
        self.output_shapes = tf.compat.v1.data.get_output_shapes(dataset)
        return dataset
