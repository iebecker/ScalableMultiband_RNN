from tensorflow import constant, string, data, io, float32, reshape, one_hot, cast, int32, sparse, int64, function, \
    math, compat
from glob import glob


class Parser:
    def __init__(self,
                 n_bands,
                 num_classes,
                 w,
                 batch_size,
                 mode,
                 num_threads,
                 physical_parameters=[],
                 physical_parameters_est=[],
                 buffer_size=10000):
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

    def parser_structure(self):
        self.context_features = {'Label': io.FixedLenFeature([], dtype=int64),
                                 'ID': io.FixedLenFeature([], dtype=string)}
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.context_features[param] = io.FixedLenFeature([], dtype=float32)
            for param in self.phys_params_est:
                self.context_features[param] = io.FixedLenFeature([], dtype=float32)

        self.sequence_features = {}
        for i in range(self.n_bands):
            self.context_features['N_' + str(i)] = io.FixedLenFeature([], dtype=int64)  # Length of lc per band
            self.context_features['M0_' + str(i)] = io.FixedLenFeature([], dtype=float32)  # First magnitude
            self.context_features['T0_' + str(i)] = io.FixedLenFeature([], dtype=float32)  # First time
            self.sequence_features['LightCurve_' + str(i)] = io.VarLenFeature(
                dtype=float32)  # Matrix representation
            self.sequence_features['Order_' + str(i)] = io.VarLenFeature(dtype=int64)  # Order of observations
            self.sequence_features['Uncertainty_' + str(i)] = io.VarLenFeature(
                dtype=float32)  # Uncertainty representation

    @function
    def log10(self, x):
        numerator = math.log(x)
        denominator = math.log(constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def get_pad_shapes_and_values(self):
        # Values of padding
        self.pad_values_X = {'ID': constant('', dtype=string)}

        # Shapes of the padding Objects
        self.padd_shapes_X = {'ID': []}

        for i in range(self.n_bands):
            self.padd_shapes_X['input_LC_' + str(i)] = [None, self.w]
            self.pad_values_X['input_LC_' + str(i)] = constant(0, dtype=float32)

            self.padd_shapes_X['O_' + str(i)] = [None]
            self.pad_values_X['O_' + str(i)] = constant(999, dtype=int32)

            self.padd_shapes_X['N_' + str(i)] = []
            self.pad_values_X['N_' + str(i)] = constant(0, dtype=int32)

            self.padd_shapes_X['M0_' + str(i)] = []
            self.pad_values_X['M0_' + str(i)] = constant(0, dtype=float32)

            self.padd_shapes_X['T0_' + str(i)] = []
            self.pad_values_X['T0_' + str(i)] = constant(0, dtype=float32)

            self.padd_shapes_X['U_' + str(i)] = [None]
            self.pad_values_X['U_' + str(i)] = constant(0, dtype=float32)

        # For the target dictionary
        self.padd_values_Y = {'Class': constant(0, dtype=int32)}
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.padd_values_Y[param] = constant(0.0, dtype=float32)

        # For the light curves
        if 'language' in self.mode:
            for i in range(self.n_bands):
                self.padd_values_Y['LC_' + str(i)] = constant(0, dtype=float32)

        # The shapes of the classifier and regressions
        self.padd_shapes_Y = {'Class': [self.num_classes, ]}
        if 'regression' in self.mode:
            for param in self.phys_params:
                self.padd_shapes_Y[param] = []
        if 'language' in self.mode:
            for i in range(self.n_bands):
                self.padd_shapes_Y['LC_' + str(i)] = [None, self.w]

    def data_parser(self, serialized_example):
        """Parse the serialized objects."""

        context_data, sequence_data = io.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=self.context_features,
            sequence_features=self.sequence_features
        )

        # Extract context features
        # Extract ID
        ID = cast(context_data['ID'], string)
        # label of the light curve
        label = cast(context_data['label'], int32)
        # Encode the label as a hone-hot tensor
        label = one_hot(label, self.num_classes, on_value=1, off_value=0, axis=-1, dtype=int32)

        # Extract regression parameters
        if 'regression' in self.mode:
            phys_params = {}
            for param in self.phys_params:
                var = cast(context_data[param], float32)
                phys_params[param] = var
            # Extract regression estimate parameters
            phys_params_est = {}
            for param in self.phys_params_est:
                var = cast(context_data[param], float32)
                phys_params_est[param] = var

        # Extract info per band
        ms = [None] * self.n_bands
        ts = [None] * self.n_bands
        ns = [None] * self.n_bands
        l_cs = [None] * self.n_bands
        os = [None] * self.n_bands
        us = [None] * self.n_bands

        for i in range(self.n_bands):
            # First observation per band
            ms[i] = cast(context_data['M0_' + str(i)], float32)
            # First time per band
            ts[i] = cast(context_data['T0_' + str(i)], float32)
            # Number of observations per band
            ns[i] = cast(context_data['N_' + str(i)], int32)
            # Extract light curve representation
            l_cs[i] = sequence_data['LightCurve_' + str(i)]
            l_cs[i] = sparse.to_dense(l_cs[i])
            l_cs[i] = cast(l_cs[i], float32)
            # Reshape the curves to its original matrix form
            l_cs[i] = reshape(l_cs[i], [ns[i], self.w])
            # Extract Order sequence
            os[i] = sequence_data['Order_' + str(i)]
            os[i] = sparse.to_dense(os[i])
            os[i] = cast(os[i], int32)
            # Reshape the order to its original form
            os[i] = reshape(os[i], [ns[i]])
            # Uncertainty containers
            us[i] = sequence_data['Uncertainty_' + str(i)]
            us[i] = sparse.to_dense(us[i])
            us[i] = cast(us[i], float32)
            us[i] = reshape(us[i], [ns[i]])

        # output dictionary
        dict_in = {'ID': ID}
        for i in range(self.n_bands):
            dict_in['input_LC_' + str(i)] = l_cs[i]
            dict_in['O_' + str(i)] = os[i]
            dict_in['N_' + str(i)] = ns[i]
            dict_in['M0_' + str(i)] = ms[i]
            dict_in['T0_' + str(i)] = ts[i]
            dict_in['U_' + str(i)] = us[i]

        dict_out = {'Class': label}
        if 'regression' in self.mode:
            for param in self.phys_params:
                dict_out[param] = phys_params[param]
        if 'language' in self.mode:
            for i in range(self.n_bands):
                dict_out['LC_' + str(i)] = l_cs[i]

        return dict_in, dict_out

    def get_dataset(self,
                    filename,
                    epochs,
                    shuffle=True):
        """
        Parse the files in filename
        """
        # If filename contains a wildcard, get all the files
        if '*' in filename:
            filename = glob(filename)

        dataset = data.TFRecordDataset(filename,
                                       num_parallel_reads=data.experimental.AUTOTUNE,
                                       )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        # Deserialize and Parse
        dataset = dataset.map(self.data_parser,
                              num_parallel_calls=data.experimental.AUTOTUNE
                              )

        # Repeat epochs times
        dataset = dataset.repeat(count=epochs)

        self.get_pad_shapes_and_values()

        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(self.padd_shapes_X, self.padd_shapes_Y),
                                       drop_remainder=False,
                                       padding_values=(self.pad_values_X, self.padd_values_Y),
                                       )
        dataset = dataset.prefetch(data.experimental.AUTOTUNE)

        self.output_types = compat.v1.data.get_output_types(dataset)
        self.output_shapes = compat.v1.data.get_output_shapes(dataset)
        return dataset
