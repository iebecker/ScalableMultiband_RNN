import tensorflow as tf


class CustomModelPhysParams(tf.keras.Model):
    def __init__(self, regression_size, physical_params, **kwargs):
        super(CustomModelPhysParams, self).__init__(**kwargs)
        self.physical_params = physical_params
        self.regression_size = regression_size

    def compute_masks(self, inputs_central):
        self.maskeds = {}
        for param in self.physical_params:
            self.maskeds[param] = tf.math.not_equal(inputs_central[param], 0.0, name='Mask_' + param)
            self.maskeds[param] = tf.cast(self.maskeds[param], tf.float32)

    def call(self):

        denses_ = {}
        for param in self.physical_params:
            denses_[param] = dense_central
            for l in range(len(self.regression_size)):
                denses_[param] = tf.keras.layers.Dense(self.regression_size[l],
                                                       activation='relu',
                                                       use_bias=True,
                                                       name='Complexity_' + param + '_' + str(l)
                                                       )(denses_[param])

        # Proyection for the loss
        outputs_ = {}
        for param in self.physical_params:
            outputs_[param] = tf.keras.layers.Dense(1,
                                                    activation=None,
                                                    use_bias=True,
                                                    name='Prediction_' + param)(denses_[param])
            outputs_[param] = tf.keras.layers.multiply(maskeds[param], outputs_[param][:, 0],
                                                       name='Masked_Prediction_' + param)


class CustomModelBand(tf.keras.Model):
    def __init__(self, signature, N_skip, **kwargs):
        super(CustomModelBand, self).__init__(**kwargs)
        self.input_signature = signature

        self.train_step = tf.function(self.train_step_temp, input_signature=self.input_signature)
        self.model_number = kwargs['name'].split('_')[1]
        self.kwargs = kwargs
        self.N_skip = N_skip

    def get_config(self):
        config = {"input_signature": self.input_signature, "model_number": self.model_number,
                  "name": self.kwargs['name']}
        return config

    def compute_weights(self, uncert):
        """Compute sample weights based on the uncertainty input"""

        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(uncert)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(uncert)[0], tf.shape(uncert)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Subtract 1 to the mask. This will ommit the first N_skip observations
        mask = 1.0 - m1

        # Apply the mask previously computed
        uncert = tf.math.multiply_no_nan(mask, uncert)

        # Compute un-normalized weights
        weights = tf.math.reciprocal_no_nan(uncert)

        # Add them to compute the norm constant
        norm_weights = tf.reduce_sum(weights, axis=1)

        # Divide the weights by the norm
        normed_weights = tf.math.divide_no_nan(weights, tf.reshape(norm_weights, (-1, 1)))

        return normed_weights

    def train_step_temp(self, input_, target_):
        """Function that trains a band-specific model"""

        sample_weight_ = self.compute_weights(input_['U_' + self.model_number])
        with tf.GradientTape() as tape:
            predictions = self(input_, training=True)['Class']
            loss_value = self.compiled_loss(y_true=target_['Class'],
                                            y_pred=predictions,
                                            sample_weight=sample_weight_)

        gradients = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target_, predictions, sample_weight_)
        # Return a dict_transform mapping metric names to current value
        out = {m.name: m.result() for m in self.metrics}
        out['train_loss'] = loss_value
        return out


class CustomModelCentral(tf.keras.Model):
    def __init__(self, signature, n_bands, N_skip, **kwargs):
        super(CustomModelCentral, self).__init__(**kwargs)
        self.input_signature = signature
        self.train_step = tf.function(self.train_step_temp, input_signature=self.input_signature)

        self.target_phys = list(self.input_signature[1].keys())
        self.target_phys.remove('FinalClass')
        self.target_phys.remove('Class')

        self.n_bands = n_bands
        self.N_skip = N_skip

    def create_mask(self, input):
        """Create a tensor to mask the first N_skip elements"""
        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(input)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(input)[0], tf.shape(input)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Subtract 1 to the mask. This will omit the first N_skip observations
        mask = 1.0 - m1
        return mask

    @staticmethod
    def sort(tensor_test, indices):
        """Sort tensor_test given the order in indices"""
        shapes = tf.shape(tensor_test, name='Get_shapes')
        M = shapes[0]
        N = shapes[1]
        X, Y = tf.meshgrid(tf.range(0, N), tf.range(0, M))
        tf_indices = tf.stack([Y, indices], axis=2)
        sorted_tensor = tf.gather_nd(tensor_test, tf_indices)
        return sorted_tensor

    def sort_uncert(self, input_states, input_orders):
        """Concatenate and sort the inputs given the order information and the
        length information (N)"""

        # Assign a band number to each element of the unput list
        code_bands = []
        for band in range(len(input_states)):
            shape = tf.shape(input_states[band])
            ones = tf.ones((shape[0], shape[1]), tf.float32)
            code_band = tf.multiply(tf.cast(band, tf.float32), ones)
            code_bands.append(code_band)

        concat_codes = tf.concat(code_bands,
                                 axis=1,
                                 name='Concat_codes')

        concat_states = tf.concat(input_states,
                                  axis=1,
                                  name='Concat_states')

        # Concatenate the orders obtained from the data
        concat_orders = tf.concat(input_orders,
                                  axis=1,
                                  name='Concat_orders')

        # Sort the concatenated orders
        sorted_concat_orders = tf.argsort(concat_orders,
                                          axis=1,
                                          direction='ASCENDING',
                                          stable=True,
                                          name='Argsort')

        # Sort the concatenated states, to be used as an input
        sorted_states = self.sort(concat_states,
                                  sorted_concat_orders)
        sorted_codes = self.sort(concat_codes,
                                 sorted_concat_orders)

        # Make the mask for the central states
        n0 = tf.cast(tf.math.not_equal(sorted_states, 0.0),
                     tf.float32)

        n_central = tf.cast(tf.reduce_sum(n0, axis=1),
                            tf.int32)
        n_max = tf.reduce_max(n_central)

        # Omit the last empy steps
        sorted_states = sorted_states[:, :n_max]
        sorted_codes = sorted_codes[:, :n_max]

        return sorted_states, sorted_codes

    def multiband_uncert(self, inputs):
        """Compute sample weights based on the uncertainty input of many bands,
        skip the first N_skip elements."""

        # Obtain the mean uncertainty for each LC for each band
        normed_uncerts = [[]] * self.n_bands
        uncerts = [[]] * self.n_bands
        orders_ = [[]] * self.n_bands

        # Extract the uncertainty per band
        for j in range(self.n_bands):
            uncerts[j] = inputs['U_' + str(j)]
            orders_[j] = inputs['O_' + str(j)]

        # Concatenate the uncertainties and return the N and the code
        concat_uncert, concat_code = self.sort_uncert(uncerts, orders_)

        # Skip N_skip values
        skip_uncert = concat_uncert[:, self.N_skip:]
        skip_code = concat_code[:, self.N_skip:]

        # Split the codes in each band, normalize them and consolidate all of them in a single tensor
        normed_uncerts = []
        for band in range(self.n_bands):
            # Split by band
            b = skip_code == band
            # Cast to float32
            float_mask = tf.cast(b, tf.float32)
            # Mask the uncertainties per band
            band_uncert = tf.math.multiply_no_nan(float_mask, skip_uncert)
            # Compute the sum of the uncertainties, to normalize
            den_uncert = tf.math.reduce_sum(band_uncert, axis=1, keepdims=True)
            # Normalize per band
            normed_uncert_band = tf.math.divide_no_nan(band_uncert, den_uncert)
            # Add them to a list in order to consolidate them
            normed_uncerts.append(normed_uncert_band)

        # Compute the sum. Since the tensors are masked, they will match the observations without having to reshape anything
        normed_uncerts = tf.add_n(normed_uncerts)

        # Return to the original shape
        # Add the skipped weights as 0
        skipped = tf.zeros([tf.shape(normed_uncerts)[0], self.N_skip])
        # Concatenate
        normed_uncerts = tf.concat([skipped, normed_uncerts], axis=1)
        return normed_uncerts

    def compute_weights(self, uncert):
        """Compute sample weights based on the uncertainty input"""

        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(uncert)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(uncert)[0], tf.shape(uncert)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Subtract 1 to the mask. This will ommit the first N_skip observations
        mask = 1.0 - m1

        # Apply the mask previously computed
        uncert = tf.math.multiply_no_nan(mask, uncert)

        # Compute un-normalized weights
        weights = tf.math.reciprocal_no_nan(uncert)

        # Add them to compute the norm constant
        norm_weights = tf.reduce_sum(weights, axis=1)

        # Divide the weights by the norm
        normed_weights = tf.math.divide_no_nan(weights, tf.reshape(norm_weights, (-1, 1)))

        return normed_weights

    def train_step_temp(self, input_, target_):
        """Function that trains a band-specific model"""

        with tf.GradientTape() as tape:
            normed_uncerts = self.multiband_uncert(input_)
            sample_weight_ = self.compute_weights(normed_uncerts)
            # Here we have the output of the model. Dict in the real case
            predictions = self(input_,
                               training=True)
            loss_value = self.compiled_loss(y_true=target_,  # ['Class'],
                                            y_pred=predictions,
                                            sample_weight=sample_weight_
                                            )
        gradients = tape.gradient(loss_value, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target_, predictions)
        # Return a dict_transform mapping metric names to current value
        out = {m.name: m.result() for m in self.metrics}
        out['train_loss'] = loss_value
        return out
