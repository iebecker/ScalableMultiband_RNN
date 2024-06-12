import tensorflow as tf
from itertools import combinations
import numpy as np


class PhysicalParamsLayer(tf.keras.Model):
    # This model behaves differently in prediction and in
    def __init__(self,
                 regression_size,
                 physical_params,
                 dropout,
                 **kwargs):
        super(PhysicalParamsLayer, self).__init__(**kwargs)
        self.supports_masking = True
        # self.n_bands = n_bands
        self.physical_params = physical_params
        self.regression_size = regression_size
        self.dropout = dropout

        # Define the Dense layers
        self.ParamsLayers = {}
        self.DensePhys = {}
        for param in self.physical_params:
            self.ParamsLayers[param] = []
            for l in range(len(self.regression_size)):
                # Add the layer
                self.ParamsLayers[param].append(tf.keras.layers.Dense(self.regression_size[l],
                                                                      activation='relu',
                                                                      use_bias=True,
                                                                      )
                                                )

            self.DensePhys[param] = tf.keras.layers.Dense(1,
                                                          activation=None,
                                                          use_bias=True,
                                                          name='Prediction_' + param)

    def call(self, dense_central):
        # Compute the mask
        PhysPred = {}
        # Compute the transformation
        for param in self.physical_params:
            output = dense_central
            for l in range(len(self.regression_size)):
                output = self.ParamsLayers[param][l](output)

            PhysPred[param] = self.DensePhys[param](output)
        return PhysPred


class MeanColorLayer(tf.keras.layers.Layer):
    def __init__(self, n_bands, **kwargs):
        super(MeanColorLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.n_bands = n_bands
        self.compute_ColorMatrix()

    def my_max(self, prev, val):
        return tf.maximum(prev, val)

    def compute_ColorMatrix(self):
        """ Obtains the matrix to compute the colors (magnitude differences)
        between all the bands. In the 2-band scenario it reduces to [[1, -1]].
        """
        NN = self.n_bands
        # Compute all possible combinations
        combs = set(combinations(np.arange(NN), 2))
        combs = [list(i) for i in combs]
        combs = tf.constant(combs)

        # Compute the index of the first elements
        v1 = tf.reshape(combs[:, 0], [-1, 1])
        v1b = tf.reshape(tf.range(tf.shape(combs)[0]), [-1, 1])
        v1 = tf.concat([v1b, v1], axis=1)
        # Compute the index of the second elements
        v2 = tf.reshape(combs[:, 1], [-1, 1])
        v1b = tf.reshape(tf.range(tf.shape(combs)[0]), [-1, 1])
        v2 = tf.concat([v1b, v2], axis=1)

        # Concatenate and cast to int64
        indices = tf.concat([v1, v2], axis=0)
        indices = tf.cast(indices, tf.int64)

        # First part are the first elements
        u1 = tf.ones(tf.shape(v1)[0])
        # Second part are the subtracted elements
        u2 = -tf.ones(tf.shape(v1)[0])
        updates = tf.concat([u1, u2], axis=0)

        # Compute the shape #combinations x bands
        shape = tf.concat([tf.shape(v1)[0], NN], axis=0)
        shape = tf.cast(shape, tf.int64)

        # Obtain the matrix
        diff_matrix = tf.scatter_nd(indices, updates, shape)

        # Transpose it to perform MatMult afterwards
        self.diff_matrix = tf.transpose(diff_matrix)

    def call(self, color, Ns, order):
        """Computes the mean mags at each timestep updating the mean colors
        as the observations arrive.
        """
        scatter = self.compute_MeanMagContainer(color, Ns, order)
        self.scatter_backup = scatter

        carry_color = [None] * self.n_bands
        for col in range(self.n_bands):
            # Identifies the unobserved bands per time-step
            mask = tf.equal(scatter[col], 0)
            # Obtains the indices of the observations
            idx = tf.where(~mask, tf.range(tf.shape(mask)[1]), 0)
            # Applies the cumulative function my_max over the time-steps
            # This line enables the replacement the empty observations in
            # scatter with the previous observations
            # The result is a sequence of (repeated) numbers,
            # where each number represents the observation to be repeated.
            idx2 = tf.transpose(tf.scan(self.my_max, tf.transpose(idx)))

            temp = tf.reshape(tf.range(tf.shape(idx2)[0]), [-1, 1])

            temp2 = tf.tile(temp, [1, tf.shape(idx2)[1]])

            indices = tf.stack([temp2, idx2], axis=0)

            indices = tf.transpose(indices)
            # Use gather to create the matrix containing the Mean Mag information
            carry_color[col] = tf.transpose(tf.gather_nd(scatter[col], indices))

        # Stack the mean mags
        carry_color = tf.stack(carry_color, axis=2)
        self.carry_color_back = carry_color
        # Multiply the entire batch
        result = tf.matmul(carry_color, self.diff_matrix)
        self.result = result
        # batch_size x time_steps x Number_colors (1 for 2 bands, 3 por 3, 6 for 4 bands)

        # N1+N2!= L1+L2
        # Cut the end of the states, those are not needed
        Ns = tf.reduce_sum(Ns, axis=0)
        N_max = tf.reduce_max(Ns)
        result = result[:, :N_max, :]

        return result

    def compute_MeanMagContainer(self, color, Ns, order):
        """Compute the mean_magnitude containers according to their observation order.
        Fills with 0's the places where it doesn't have observations according
        to the order of observations."""
        # Length of multiband timeseries
        N = tf.reduce_max(tf.reduce_sum(Ns, axis=0))

        # Pad colors to the max of the batch in all colors.
        for i in range(self.n_bands):
            color[i] = tf.pad(color[i], [[0, 0], [0, N - tf.shape(color[i])[1]]])

            # Padd the orders
            order[i] = tf.pad(order[i], [[0, 0], [0, N - tf.shape(order[i])[1]]], constant_values=999)
            order[i] = tf.cast(order[i], tf.int64)

        # Create the indices for the 0th dimension
        val1 = tf.repeat(tf.reshape(tf.range(tf.shape(order[0])[0]), [-1, 1]), repeats=[N], axis=1)
        # Cast it to int64 to use in scatter_nd
        val1 = tf.cast(val1, tf.int64)

        # Compute the assignment
        scatter = [None] * self.n_bands
        for col in range(self.n_bands):
            indices = tf.stack([val1, order[col]], axis=2)

            shape = tf.shape(color[col])
            shape = tf.cast(shape, tf.int64)

            # Perform the assignment
            scatter[col] = tf.scatter_nd(indices, color[col], shape)
        return scatter


class RawTimesLayer(tf.keras.layers.Layer):
    def __init__(self, w, **kwargs):
        super(RawTimesLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.time_col = int(w / 2) - 1

    def call(self, data, batch_N, batch_T0):
        """Compute the raw times given the delta times
        We return all except the first time.
        We mask N+self.time_col to account for the first elements of the times.
        They will be cleaned in when we merge all the times."""

        # Get the longest series
        N = tf.reduce_max(batch_N)
        # Get the time 0 for each element.
        # Expand dims to transform it into column vector
        t0 = tf.expand_dims(batch_T0, 1)

        # Create a ones matrix with N+1 dimensions.
        # The transformation from delta to time increases it by one
        #  since we have to include the first s deltas
        # We create a 1 filled matrix
        ones = tf.ones((N + self.time_col, N + self.time_col),
                       dtype=tf.float32)
        # And remove the uppper part
        tri_lower = tf.linalg.band_part(ones,
                                        num_lower=-1,
                                        num_upper=0,
                                        )
        # We expand the dims to repeat the matrix
        tri_lower = tf.expand_dims(tri_lower, axis=0)
        # We replicate the lower triangual matrix the batch times (it can be different if we reach the end of the dataset)
        tri_lower = tf.tile(tri_lower, [tf.shape(batch_N)[0], 1, 1])

        ## Get the last column of delta times
        # t2-t1
        # t3-t2
        # tN - tN-1
        data1 = data[:, :, self.time_col]

        # # Get the first deltas from the first element of the "w-1" columns
        data0 = data[:, 0, :self.time_col]

        # # Insert the first deltas into the first positions of the columns of deltas
        data_ext = tf.concat([data0, data1], axis=1)

        # Compute 10^log(delta time)
        data_ext = tf.math.pow(10.0, data_ext)

        # Compute the times from deltas
        # Do matrix times vector and add the t0
        times = tf.linalg.matvec(tri_lower, data_ext) + t0

        # Ommit the first time_col -1 elements, as those are not used
        # the -1 is used to compute the general time differences
        ini = tf.maximum(self.time_col - 1, 0)
        times = times[:, ini:]

        # Remove the unwanted times from the vectors
        # +1 stands for the extra time we add
        masks = tf.sequence_mask(batch_N + 1)
        masks = tf.cast(masks, tf.float32)  # MAsk to float.
        masked_time = tf.multiply(times, masks)

        # Put the times as 9999, so they are palced at the end when we sort everything
        masks2 = 1.0 - masks
        masks2 = 999.99 * masks2

        # Add the values
        masked_time = masked_time + masks2
        return masked_time


class AllTimes(tf.keras.layers.Layer):
    def __init__(self,
                 n_bands,
                 **kwargs
                 ):
        super(AllTimes, self).__init__(**kwargs)
        # self.num_classes = num_classes
        self.supports_masking = True
        self.n_bands = n_bands

    def call(self, input_times, N_total):
        sorted_times = self.sort_times(input_times)
        # Get the longest series
        N = tf.reduce_max(N_total)

        # N1+N2!= L1+L2
        # The longest in one band isn't the longest in the other.
        # We crop up to the longest sequence
        sorted_times = sorted_times[:, :(N + 1)]

        # Create a matrix to compute the differences
        ones = tf.ones((N, N))
        diag = tf.linalg.band_part(ones, 0, 0)
        top = tf.linalg.band_part(ones, 1, 0)
        delta = -top + 2 * diag
        # Add the first column
        fc = -diag[:, :1]
        # Concatenate it with the matrix
        delta = tf.concat((fc, delta), axis=1, name='AllTimes_add_fc')
        # We expand the dims to repeat the matrix
        delta = tf.expand_dims(delta, axis=0)
        # We replicate the matrix delta batch size times
        delta = tf.tile(delta, [tf.shape(N_total)[0], 1, 1])
        # Compute the time differences
        diffs = tf.linalg.matvec(delta, sorted_times)

        # Remove the unwanted times from the vectors
        masks = tf.sequence_mask(N_total)
        masks = tf.cast(masks, tf.float32)  # MAsk to float.
        masked_diffs = tf.multiply(diffs, masks)

        # Expand the dimensions to concatenate them to the hidden state
        masked_diffs = tf.expand_dims(masked_diffs, 2)
        return masked_diffs

    def sort_times(self, input_times):
        """
        Sort the input_times list accordingto the input_orders list.
        Each list contains n_bands elements.
        Computes the total length of the final sequences.
        """
        concat_times = tf.concat(input_times,
                                 axis=1,
                                 name='Concat_times')

        # Sort the concatenated times
        sorted_times = tf.sort(concat_times,
                               axis=1,
                               direction='ASCENDING',
                               name='sorted_times')
        # Remove the first n_bands-1 elements
        sorted_times = sorted_times[:, self.n_bands - 1:]

        return sorted_times


class MeanMagLayer(tf.keras.layers.Layer):
    def __init__(self, w, **kwargs):
        super(MeanMagLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.w = w

    def call(self, data, batch_N, batch_M0):
        """Compute the mean color per step given the delta mags"""

        # Get the longest series
        N = tf.reduce_max(batch_N)
        # Get the mag 0 for each element.
        # Expand dims to transform it into column vector
        m0 = tf.expand_dims(batch_M0, 1)

        # Create a ones matrix with N+1 dimensions.
        # The transformation from delta to mag increases it by one
        # We create a 1 filled matrix
        ones = tf.ones((N + 1, N + 1),
                       dtype=tf.float32)
        # And remove the uppper part
        tri_lower = tf.linalg.band_part(ones,
                                        num_lower=-1,
                                        num_upper=0,
                                        )
        # We expand the dims to repeat the matrix
        tri_lower = tf.expand_dims(tri_lower, axis=0)
        # We replicate the lower triangual matrix the batch times (it can be different if we reach the end of the dataset)
        tri_lower = tf.tile(tri_lower, [tf.shape(batch_N)[0], 1, 1])

        # Get the last column of deltas
        # m2-m1
        # m3-m2
        # mN - mN-1
        # Assuming step s = 1
        mag_col = int(self.w / 2)
        data1 = data[:, :, -1]

        # Get the first delta from the first element of the third column
        data0 = data[:, :1, mag_col]

        # Insert the first delta into the first positions of the columns of deltas
        data_ext = tf.concat([data0, data1], axis=1)

        # Compute the magnitudes from deltas
        # Do matrix times vector and add the mag0
        mags = tf.linalg.matvec(tri_lower, data_ext) + m0

        # Add the magnitudes at each step to get the updated mean
        sum_mags = tf.linalg.matvec(tri_lower, mags) + m0
        # Select all but the first one as we have 1 more datapoint
        sum_mags = sum_mags[:, 1:]

        # Divide the sum of mags by the number of observations
        # The sequence starts with m0+m1+m2, so we divide by 3 first
        N_obs = tf.range(3, N + 3, name='N_obs', dtype=tf.float32)

        # Compute the mean per time-step
        mean_mags = tf.divide(sum_mags, N_obs)
        return mean_mags


class LastRelevantLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LastRelevantLayer, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, output, length):
        """Get the last relevant output from the network"""
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = tf.shape(output)[2]

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


class SauceLayer(tf.keras.layers.Layer):
    def __init__(self, shape, **kwargs):
        super(SauceLayer, self).__init__(**kwargs)
        self.supports_masking = True
        self.shape = shape

    def build(self, input_shape):
        self.scale = tf.Variable([1 / self.shape for _ in range(self.shape)], trainable=True)

    def call(self, inputs):
        # Softmax normalized
        scale = tf.nn.softmax(self.scale)
        return tf.tensordot(scale, inputs, axes=1)


class ApplyMask(tf.keras.layers.Layer):
    def __init__(self, num_classes, mask_value=-99.99, **kwargs):
        super(ApplyMask, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.supports_masking = True
        self.mask_value = mask_value

    def call(self, inputs, N):
        masks = tf.sequence_mask(N)
        masks = tf.cast(masks, tf.float32)  # MAsk to float.
        masks = tf.expand_dims(masks, 2)  # Add third dimension
        mask_expand = tf.concat(self.num_classes * [masks], axis=2)  # Expand mask along class dimension

        # N1+N2!= L1+L2
        # The longest in one band isn't the longest in the other.
        # We remove the final columns which have no information
        # Obtain Cropped inputs
        max_N = tf.math.reduce_max(N)
        cropped_inputs = inputs[:, :max_N, :]
        # Masked elements zero
        masked_prediction = tf.multiply(cropped_inputs, mask_expand)  #

        # Invert the values of the mask
        mask3 = 1.0 - mask_expand  # 0 0 0 0 1 1 1 1
        mask3 = self.mask_value * mask3

        # Add the padding new values
        masked_prediction = masked_prediction + mask3  # v1 v2 v3 -99.99 -99.99
        return masked_prediction


class InputCentral(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InputCentral, self).__init__(**kwargs)
        # self.num_classes = num_classes
        self.supports_masking = True

    @tf.function(reduce_retracing=True)
    def sort_states(self, tensor_test, indices):
        """Sort tensor_test given the order in indices"""
        shapes = tf.shape(tensor_test, name='Get_shapes')
        M = shapes[0]
        N = shapes[1]
        X, Y = tf.meshgrid(tf.range(0, N), tf.range(0, M))
        tf_indices = tf.stack([Y, indices], axis=2)
        sorted_tensor = tf.gather_nd(tensor_test, tf_indices)
        return sorted_tensor

    def call(self, input_states, input_orders, input_Ns):
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
        sorted_states = self.sort_states(concat_states, sorted_concat_orders)

        # Make the mask for the central states
        N_central = tf.reduce_sum(input_Ns,
                                  axis=0,
                                  name='Central_length')

        # N1+N2!= L1+L2
        # Cut the end of the states, those are not needed
        N_max = tf.reduce_max(N_central)
        sorted_states = sorted_states[:, :N_max, :]

        return sorted_states, N_central


class RNNLayersCentral(tf.keras.Model):
    def __init__(self,
                 hidden_sizes,
                 implementation=1,
                 return_sequences=True,
                 return_state=False,
                 bidirectional=False,
                 **kwargs
                 ):
        super(RNNLayersCentral, self).__init__(**kwargs)
        self.supports_masking = True
        self.bidirectional = bidirectional
        self.rnns = [[], []]
        # Define the parameters of the RNNs
        cudnn_kwargs = {'activation': 'tanh',
                        'recurrent_activation': 'sigmoid',
                        'recurrent_dropout': 0,
                        'unroll': False,
                        'use_bias': True,
                        'stateful': False,
                        'return_sequences': True,
                        'return_state': False
                        }

        # Create the proyection for the first layer of the residual connection
        self.proyection_Wf = tf.keras.layers.Dense(hidden_sizes[0],
                                                   use_bias=False,
                                                   activation=None,
                                                   name='Res_W_f')
        self.LayerNorm_f = [tf.keras.layers.LayerNormalization() for _ in range(len(hidden_sizes) - 1)]

        if self.bidirectional:
            self.proyection_Wb = tf.keras.layers.Dense(hidden_sizes[0],
                                                       use_bias=False,
                                                       activation=None,
                                                       name='Res_W_b')
            self.LayerNorm_b = [tf.keras.layers.LayerNormalization() for _ in range(len(hidden_sizes) - 1)]

        # Simplest mode, forward RNNs
        if not bidirectional:
            # This is to have the same structure in the bidir case
            for i in range(len(hidden_sizes)):
                RNN = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                           **cudnn_kwargs,
                                           name='LSTM_' + str(i) + '_Central')

                self.rnns[0].append(RNN)

        # Bidirectional RNNs
        elif bidirectional:
            # Satisfy the conditions to use cudnn kernel
            for i in range(len(hidden_sizes)):
                cudnn_kwargs['go_backwards'] = True
                RNN_b = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                             **cudnn_kwargs,
                                             name='LSTM_b' + str(i) + '_Central')
                cudnn_kwargs['go_backwards'] = False
                RNN_f = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                             **cudnn_kwargs,
                                             name='LSTM_f' + str(i) + '_Central')
                self.rnns[0].append(RNN_f)
                self.rnns[1].append(RNN_b)

    def call(self, input_, N_, training=True):
        """Excecute the rnns based on the input.
        outputs the last states of the last layer."""
        # Get the boolean mask
        mask_ = tf.sequence_mask(N_, )

        # Create the output container
        outputs_f = []
        # Append the output of the first RNN
        output_f = self.rnns[0][0](inputs=input_,
                                   mask=mask_
                                   )
        outputs_f.append(output_f)

        # Compute the residual to be used as input of the next RNNs
        residual = output_f + self.proyection_Wf(input_)
        residual = self.LayerNorm_f[0](residual)
        # print(list(range(1, len(self.rnns[0]))))
        for i in range(1, len(self.rnns[0])):
            # print(i)
            # Input the residual
            output_f = self.rnns[0][i](inputs=residual,
                                       mask=mask_
                                       )
            outputs_f.append(output_f)
            # print(len(self.rnns[0]), len(self.rnns[0])-1, i == len(self.rnns[0])-1)
            # We don't compute the last residual
            if i == len(self.rnns[0]) - 1:
                break

            # Add the residual connection
            residual = output_f + outputs_f[-1]
            residual = self.LayerNorm_f[i](residual)

        # Flip the output from the backwards RNN if applicable
        if self.bidirectional:
            # Create the output container
            outputs_b = []
            # Append the output of the first RNN
            output_b = self.rnns[1][0](inputs=input_,
                                       mask=mask_,
                                       # go_backwards=True
                                       )
            outputs_b.append(output_b)

            # Compute the residual to be used as input of the next RNNs
            residual = output_b + self.proyection_Wb(input_)
            # residual = self.LayerNorm_b[0](residual)
            for i in range(1, len(self.rnns[1])):
                # Input the residual
                output_b = self.rnns[1][i](inputs=residual,
                                           mask=mask_,
                                           # go_backwards=True
                                           )
                outputs_b.append(output_b)

                # We don't compute the last residual
                if i == len(self.rnns[1]) - 1:
                    break

                # Add the residual connection
                residual = output_b + outputs_b[-1]
                residual = self.LayerNorm_b[i](residual)

            # Concatenate the forward and backward outputs at the same recurrence level
            outputs = tf.concat([outputs_f, outputs_b], axis=2, name='Concat_bidir_states')
            outputs = []
            for i in range(len(self.rnns[0])):
                out = tf.concat([outputs_f[i], outputs_b[i]], axis=2)
                outputs.append(out)
            return outputs

        return outputs_f


class RNNLayersBands(tf.keras.Model):
    def __init__(self,
                 hidden_sizes,
                 index,
                 common_kernel_layer=None,
                 common_recurrent_kernel_layer=None,
                 use_mod_cell=False,
                 implementation=1,
                 return_sequences=True,
                 return_state=False,
                 bidirectional=False,
                 use_gated_common=False,
                 l1=0.0,
                 l2=0.0,
                 **kwargs
                 ):
        super(RNNLayersBands, self).__init__(**kwargs)

        """Creates RNNs for each band. It can be implemented with the custom GRU implementation
        the CUDnn implementations."""

        self.bidirectional = bidirectional
        self.use_gated_common = use_gated_common
        self.return_state = return_state
        self.use_mod_cell = use_mod_cell
        self.common_recurrent_kernel_layer = common_recurrent_kernel_layer
        self.common_kernel_layer = common_kernel_layer
        self.index = index
        self.hidden_sizes = hidden_sizes

        self.l1 = l1
        self.l2 = l2
        # Satisfy the conditions to use cudnn kernel
        cudnn_kwargs = {'activation': 'tanh',
                        'recurrent_activation': 'sigmoid',
                        'use_bias': True,
                        'implementation': implementation,
                        'return_sequences': return_sequences
                        }
        if use_mod_cell:
            cudnn_kwargs.pop('return_sequences')

        if not bidirectional:
            rnns = [None] * len(hidden_sizes)

            for i in range(len(hidden_sizes)):
                if use_mod_cell:
                    pass
                    # gru_cell = ModGRUCell(units=hidden_sizes[i],
                    #                       common_kernel_layer=common_kernel_layer[i],
                    #                       common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
                    #                       kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                    #                       bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                    #                       name='ModGRUCell' + str(i) + '_' + str(index),
                    #                       use_gated_common=use_gated_common,
                    #                       **cudnn_kwargs,
                    #                       )
                    # rnns[i] = tf.keras.layers.RNN(gru_cell,
                    #                               return_sequences=return_sequences,
                    #                               return_state=return_state,
                    #                               unroll=False,
                    #                               stateful=False,
                    #                               name='RNN' + str(i) + '_' + str(index),
                    #                               )
                elif not use_mod_cell:
                    cudnn_kwargs['recurrent_dropout'] = 0
                    cudnn_kwargs['unroll'] = False
                    cudnn_kwargs['stateful'] = False
                    cudnn_kwargs['return_sequences'] = return_sequences
                    rnns[i] = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                                   name='LSTM' + str(i) + '_' + str(index),
                                                   **cudnn_kwargs
                                                   )

        elif bidirectional:
            rnns = [[], []]
            if use_mod_cell:
                pass
                # rnns = [[],[]]
                # Satisfy the conditions to use cudnn kernel
                # cudnn_kwargs = {'activation': 'tanh', 'recurrent_activation': 'sigmoid', 'use_bias': True,
                #                 'reset_after': True, 'implementation': implementation,
                #                 'return_sequences': return_sequences}
                # # Add other parameters
                # for i in range(len(hidden_sizes)):
                #     gru_cell_f = ModGRUCell(units=hidden_sizes[i],
                #                             common_kernel_layer=common_kernel_layer[i],
                #                             common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
                #                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                #                             bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                #                             name='GRUCell' + str(i) + '_' + str(index),
                #                             use_gated_common=use_gated_common,
                #                             **cudnn_kwargs,
                #                             )
                #
                #     rnns_f = tf.keras.layers.RNN(gru_cell_f,
                #                                  return_sequences=return_sequences,
                #                                  return_state=return_state,
                #                                  unroll=cudnn_kwargs['unroll'],
                #                                  stateful=cudnn_kwargs['stateful'],
                #                                  name='RNN_f_' + str(i) + '_' + str(index),
                #                                  go_backwards=False
                #                                  )
                #
                #     gru_cell_b = ModGRUCell(units=hidden_sizes[i],
                #                             common_kernel_layer=common_kernel_layer[i],
                #                             common_recurrent_kernel_layer=common_recurrent_kernel_layer[i],
                #                             kernel_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                #                             bias_regularizer=tf.keras.regularizers.L1L2(l1=self.l1, l2=self.l2),
                #                             name='GRUCell' + str(i) + '_' + str(index),
                #                             use_gated_common=use_gated_common,
                #                             **cudnn_kwargs,
                #                             )
                #     rnns_b = tf.keras.layers.RNN(gru_cell_b,
                #                                  return_sequences=cudnn_kwargs['return_sequences'],
                #                                  return_state=return_state,
                #                                  unroll=cudnn_kwargs['unroll'],
                #                                  stateful=cudnn_kwargs['stateful'],
                #                                  name='RNN_b_' + str(i) + '_' + str(index),
                #                                  go_backwards=True
                #                                  )
                #     rnns[0].append(rnns_f)
                #     rnns[1].append(rnns_b)

            elif not use_mod_cell:
                for i in range(len(hidden_sizes)):
                    rnns_f = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                                  name='RNN_f_' + str(i) + '_' + str(index),
                                                  go_backwards=False,
                                                  **cudnn_kwargs
                                                  )
                    rnns_b = tf.keras.layers.LSTM(units=hidden_sizes[i],
                                                  name='RNN_b_' + str(i) + '_' + str(index),
                                                  go_backwards=True,
                                                  **cudnn_kwargs
                                                  )
                    rnns[0].append(rnns_f)
                    rnns[1].append(rnns_b)

        self.rnns = rnns

        # Create the protections of the residual connection
        if self.bidirectional:
            self.projection_W_f = tf.keras.layers.Dense(rnns[0][0].cell.units, name='Res_W_b_0')
            self.LayerNorm_f = [tf.keras.layers.LayerNormalization() for _ in range(len(rnns[0]) - 1)]

            self.proyection_W_b = tf.keras.layers.Dense(rnns[0][0].cell.units, name='Res_W_f_0')
            self.LayerNorm_b = [tf.keras.layers.LayerNormalization() for _ in range(len(rnns[1]) - 1)]

        else:
            self.projection_W_f = tf.keras.layers.Dense(rnns[0].cell.units, name='Res_W_b_0')
            self.LayerNorm_f = [tf.keras.layers.LayerNormalization() for _ in range(len(rnns) - 1)]

    def call(self, input_, N_):
        if self.bidirectional:
            outputs_f = self.call_one(self.rnns[0],
                                      input_,
                                      N_,
                                      backwards=False,
                                      )
            outputs_b = self.call_one(self.rnns[1],
                                      input_,
                                      N_,
                                      backwards=True,
                                      )
            self.outputs_f = outputs_f
            self.outputs_b = outputs_b

            outputs = [tf.keras.layers.concatenate([i, j],
                                                   axis=2,
                                                   )
                       for i, j in zip(outputs_f, outputs_b)]
        else:
            outputs = self.call_one(self.rnns,
                                    input_,
                                    N_
                                    )
        return outputs

    def call_one(self, rnns, input_, N_, backwards=False):
        """Execute the rnns based on the input.
        outputs the last states of the last layer."""

        # I have to propagate masks
        mask_ = tf.sequence_mask(N_)

        outputs = []
        output = rnns[0](inputs=input_, mask=mask_)
        outputs.append(output)

        if not backwards:
            proyection_W = self.projection_W_f
            LayerNorm = self.LayerNorm_f

        elif backwards:
            proyection_W = self.proyection_W_b
            LayerNorm = self.LayerNorm_b

        residual = output + proyection_W(input_)
        residual = LayerNorm[0](residual)

        for i in range(1, len(rnns)):
            # Input the residual
            output = rnns[i](inputs=residual, mask=mask_)
            outputs.append(output)
            # We don't apply LayerNorm in the final RNN layer.
            if i == len(rnns) - 1:
                break

            # Add the residual connection
            residual = output + outputs[-1]
            residual = LayerNorm[i](residual)

        # Flip the output from the backwards RNN if applicable
        if backwards:
            for j in range(len(rnns)):
                outputs[j] = tf.reverse(outputs[i], [1])

        return outputs
