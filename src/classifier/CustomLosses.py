import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, N_skip, gamma, class_weight=None, **kwargs):
        super(CategoricalFocalLoss, self).__init__(**kwargs)

        # Process focusing parameter
        gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
        self.gamma = gamma
        self.N_skip = N_skip
        self.mask_value = -99.99
        # Process class weight
        self.class_weight = class_weight
        if class_weight is not None:
            self.class_weight = tf.convert_to_tensor(class_weight,
                                                     dtype=tf.dtypes.float32)

    def __call__(self,
                 y_true,
                 y_pred,
                 sample_weight=None,
                 ):
        # Repeat y_true to match y_pred
        mask = tf.greater(y_pred[:, :, 0], self.mask_value + 1)
        mask = tf.cast(mask, tf.float32)

        # # Get the length of each sequence
        N = tf.reduce_sum(mask, axis=1)

        # Mask always the first N_skip steps

        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(mask)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Substract the first steps to the real mask
        mask = mask - m1

        # Find the number of timesteps
        reps = tf.shape(mask)[1]
        # Repeat the label along the time dimension (1)
        y_true = tf.expand_dims(y_true, 1)
        y_true = tf.repeat(y_true, [reps], axis=1)

        # If not using class weight, all sequences are equal.
        # They do not depend on the length of the LC.
        if self.class_weight is not None:
            weighted = tf.multiply(tf.cast(y_true, tf.float32), class_weight)

            loss = tf.keras.losses.categorical_crossentropy(weighted,
                                                            y_pred,
                                                            )
        else:
            loss = tf.keras.losses.categorical_crossentropy(y_true,
                                                            y_pred,
                                                            )
        # We select only the relevant prediction (sparse scenario no smoothing)
        p_hat = tf.multiply(tf.cast(y_true, tf.float32), y_pred)
        p_hat = tf.reduce_max(p_hat, axis=2)

        focal_modulation = tf.math.pow(1.0 - p_hat, self.gamma)
        loss = tf.multiply(loss, focal_modulation)

        # Apply the mask
        loss = tf.multiply(loss, mask)

        # Multiply by the weights of each lc
        if sample_weight is not None:
            loss = tf.multiply(loss, sample_weight)

            # Add over the light curve
            loss = tf.reduce_sum(loss, axis=1)

        # mean over the batch
        loss = tf.reduce_mean(loss)
        return loss


class CrossEntropy_FullWeights(tf.keras.losses.Loss):
    def __init__(self,
                 N_skip=5,
                 mask_value=0,
                 **kwargs):
        super(CrossEntropy_FullWeights, self).__init__(**kwargs)
        self.N_skip = N_skip
        self.mask_value = mask_value

    def __call__(self,
                 y_true,
                 y_pred,
                 sample_weight=None):
        mask = tf.greater(y_pred[:, :, 0], self.mask_value + 1)
        mask = tf.cast(mask, tf.float32)

        # # Get the length of each sequence
        N = tf.reduce_sum(mask, axis=1)

        # Mask always the first N_skip steps
        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(mask)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Subtract the first steps to the real mask
        mask = mask - m1

        # Find the number of timesteps
        reps = tf.shape(mask)[1]
        # Repeat the label along the time dimension (1)
        y_true = tf.expand_dims(y_true, 1)
        y_true = tf.repeat(y_true, [reps], axis=1)

        values = tf.keras.losses.categorical_crossentropy(y_true,
                                                          y_pred,
                                                          from_logits=False,
                                                          label_smoothing=0.0,
                                                          )

        # Multiply by the float32 mask
        values = tf.multiply(values, mask)

        # Multiply by the weights of each lc
        if sample_weight is not None:
            values = tf.multiply(values, sample_weight)

            # Add over the light curve
            values = tf.reduce_sum(values, axis=1)
        # mean over the batch
        values = tf.reduce_mean(values)
        return values


class MSE_masked(tf.keras.losses.Loss):
    def __init__(self, mask_value, **kwargs):
        super(MSE_masked, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, y_true, y_pred, sample_weight=None):
        # Create a loss function that adds the MSE loss (or the arguments of the function)
        # Create mask for y_true
        mask = tf.greater(y_true, self.mask_value + 1)  # Values with measurements
        mask = tf.cast(mask, tf.float32)

        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)

        # Compute raw mse
        raw_mse = tf.keras.losses.MSE(y_true_masked, y_pred_masked, )

        # Multiply by the batch size and divide by the effective number of phys params
        N_eff = tf.cast(tf.reduce_sum(mask), tf.float32)
        N = tf.cast(tf.shape(y_true)[0], tf.float32)
        mse = raw_mse * N / N_eff
        return mse
