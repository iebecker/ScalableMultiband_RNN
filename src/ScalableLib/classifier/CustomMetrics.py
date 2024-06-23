import tensorflow as tf
from tensorflow.python.ops import math_ops, nn
from tensorflow.python.keras import backend


def last_relevant(output, length):
    """Get the last relevant output from the network"""
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant


class Masked_R2(tf.keras.metrics.Metric):
    def __init__(self,
                 mask_value=-99.99,
                 name='Masked_R2'):
        super(Masked_R2, self).__init__(name)
        self.mask_value = mask_value

        self.n_ymean = self.add_weight(name='n_ymean', initializer='zeros')
        self.n = self.add_weight(name='n', initializer='zeros')
        self.SS_res = self.add_weight(name='SS_res', initializer='zeros')

        self.sum_y2 = self.add_weight(name='sum_y2', initializer='zeros')
        self.sum_y = self.add_weight(name='sum_y', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.greater(y_true, self.mask_value + 1)  # Values with measurements
        mask = tf.cast(mask, tf.float32)

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        n_ymean = tf.reduce_sum(y_true)
        n = tf.reduce_sum(mask)

        SS_res = tf.reduce_sum(tf.math.squared_difference(y_pred, y_true))

        sum_y2 = tf.reduce_sum(tf.math.square(y_true))
        sum_y = tf.reduce_sum(y_true)

        self.n_ymean.assign_add(n_ymean)
        self.n.assign_add(n)
        self.SS_res.assign_add(SS_res)

        self.sum_y2.assign_add(sum_y2)
        self.sum_y.assign_add(sum_y)

    def result(self):
        ymean = self.n_ymean / self.n
        SS_tot = self.sum_y2 - 2.0 * ymean * self.sum_y + self.n * tf.math.square(ymean)

        return 1 - tf.math.divide_no_nan(self.SS_res, SS_tot)


class Masked_RMSE(tf.keras.metrics.Metric):
    def __init__(self,
                 mask_value=-99.99,
                 name='Masked_RMSE'):
        super(Masked_RMSE, self).__init__(name)
        self.mask_value = mask_value

        self.SE = self.add_weight(name='se', initializer='zeros')
        self.N = self.add_weight(name='N', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.greater(y_true, self.mask_value + 1)  # Values with measurements
        mask = tf.cast(mask, tf.float32)

        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        error_sq = tf.reduce_sum(tf.math.squared_difference(y_pred, y_true))
        N = tf.reduce_sum(mask)

        self.SE.assign_add(error_sq)
        self.N.assign_add(N)

    def result(self):
        return tf.math.sqrt(tf.math.divide_no_nan(self.SE, self.N))


class CustomFinalF1Score(tf.keras.metrics.Metric):
    def __init__(self,
                 num_classes,
                 mask_value,
                 **kwargs):
        """Only returns macro average"""
        super(CustomFinalF1Score, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.num_classes = num_classes
        self.tp = self.add_weight("tp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fp = self.add_weight("fp", shape=(num_classes,), initializer="zeros", dtype=tf.float32)
        self.fn = self.add_weight("fn", shape=(num_classes,), initializer="zeros", dtype=tf.float32)

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Get last prediction and transform to label
        y_true, y_pred = self.RelevantPredictions(y_true, y_pred)

        # compute confussion matrix
        cm = tf.math.confusion_matrix(y_true,
                                      y_pred,
                                      num_classes=self.num_classes
                                      )

        # TP - Diagonal
        tp = tf.linalg.diag_part(cm)
        # FP - Sum over column minus diagonal
        fp = tf.reduce_sum(cm, axis=0) - tp
        # FN - Sum over row minus diagonal
        fn = tf.reduce_sum(cm, axis=1) - tp

        tp = tf.cast(tp, tf.float32)
        fp = tf.cast(fp, tf.float32)
        fn = tf.cast(fn, tf.float32)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        f1 = tf.math.divide_no_nan(self.tp, (self.tp + (self.fp + self.fn) / 2))
        macro_f1 = tf.reduce_mean(f1)  # MAcro avg
        return macro_f1


    def RelevantPredictions(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Boolean mask
        mask = tf.greater(y_pred, self.mask_value + 1)[:, :, 0]
        mask = tf.cast(mask, tf.float32)
        # Length of each lc
        N = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)
        # Extract the final prediction
        y_pred = last_relevant(y_pred, N)
        # Get the predictions via argmax
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        return y_true, y_pred


class CustomTopKFinalAccuracy(tf.keras.metrics.Metric):
    def __init__(self,
                 k=2,
                 mask_value=-99.9,
                 **kwargs):
        super(CustomTopKFinalAccuracy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.N = self.add_weight("N_batch", shape=(), initializer="zeros", dtype=tf.float32)
        self.topk = self.add_weight("Batch_TopK", shape=(), initializer="zeros", dtype=tf.float32)
        self.total_topk = self.add_weight("TopK", shape=(), initializer="zeros", dtype=tf.float32)
        self.k = k

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        topk, N = self.compute_topk(y_true, y_pred, k=self.k)
        self.N.assign_add(N)
        self.topk.assign_add(topk)

        self.total_topk = self.topk / self.N
        return self.total_topk

    def result(self):
        return self.total_topk

    def last_relevant(self, output, length):
        """Get the last relevant output from the network"""
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @tf.function(reduce_retracing=True)
    def compute_topk(self, y_true, y_pred, k=5):
        # Boolean mask
        mask = tf.greater(y_pred, self.mask_value + 1)[:, :, 0]
        mask = tf.cast(mask, tf.float32)
        # Length of each lc
        N = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)
        # Get last prediction
        y_pred = self.last_relevant(y_pred, N)

        N_batch = tf.shape(y_pred)[0]
        num_classes = tf.shape(y_pred)[1]

        true = tf.cast(math_ops.argmax(y_true, axis=-1), tf.int32)
        pred = y_pred

        res = nn.in_top_k(pred, true, k)
        values = math_ops.cast(res, backend.floatx())

        N_batch = tf.cast(N_batch, tf.float32)
        return tf.reduce_sum(values), N_batch


class CustomFinalAccuracy(tf.keras.metrics.Metric):
    def __init__(self,
                 mask_value=-99.9,
                 **kwargs):
        super(CustomFinalAccuracy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.N = self.add_weight("N_batch", shape=(), initializer="zeros", dtype=tf.float32)
        self.acc = self.add_weight("Batch_Accuracy", shape=(), initializer="zeros", dtype=tf.float32)
        self.total_acc = self.add_weight("Accuracy", shape=(), initializer="zeros", dtype=tf.float32)

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        acc, N = self.compute_acc(y_true, y_pred)
        self.N.assign_add(N)
        self.acc.assign_add(acc)

        self.total_acc = self.acc / self.N
        return self.total_acc

    def result(self):
        return self.total_acc

    def last_relevant(self, output, length):
        """Get the last relevant output from the network"""
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @tf.function(reduce_retracing=True)
    def compute_acc(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Boolean mask
        mask = tf.greater(y_pred, self.mask_value + 1)[:, :, 0]
        mask = tf.cast(mask, tf.float32)
        # Length of each lc
        N = tf.cast(tf.reduce_sum(mask, axis=1), tf.int32)

        # Extract the final prediction
        y_pred = self.last_relevant(y_pred, N)

        values = math_ops.cast(
            math_ops.equal(
                math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
            backend.floatx()
        )

        # Divide the accuracy by each sequence length - N_skip
        N_batch = tf.cast(tf.shape(mask)[0], tf.float32)

        return tf.reduce_sum(values), N_batch


class CustomAccuracy(tf.keras.metrics.Metric):
    def __init__(self,
                 N_skip:int,
                 num_classes:int,
                 mask_value:float=-99.9,
                 **kwargs):
        super(CustomAccuracy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.N_skip = N_skip
        self.num_classes = num_classes

        # Wrap the compute_acc function as a tf.function
        self.compute_signature()
        self.compute_acc = tf.function(func=self.compute_acc,
                                       input_signature=self.signature
                                       )

        self.N = self.add_weight("N_batch", shape=(), initializer="zeros", dtype=tf.float32)
        self.acc = self.add_weight("Batch_Accuracy", shape=(), initializer="zeros", dtype=tf.float32)
        self.total_acc = self.add_weight("Accuracy", shape=(), initializer="zeros", dtype=tf.float32)

    def compute_signature(self)->None:
        """Define the input signature for the compute_acc method"""
        self.signature = (tf.TensorSpec(shape=[None, self.num_classes], dtype=tf.int32),
                            tf.TensorSpec(shape=[None, None, self.num_classes], dtype=tf.float32)
                        )
    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        acc, N = self.compute_acc(y_true, y_pred)
        self.N.assign_add(N)
        self.acc.assign_add(acc)

        self.total_acc = self.acc / self.N
        return self.total_acc

    def result(self):
        return self.total_acc

    # @tf.function(reduce_retracing=True)
    # @tf.function(input_signature=(tf.TensorSpec(shape=[None, None, self.num_classes], dtype=tf.float32),
    #                               tf.TensorSpec(shape=[None, self.num_classes], dtype=tf.int32)))
    def compute_acc(self, y_true, y_pred):
        N_skip = self.N_skip
        y_true = tf.cast(y_true, tf.float32)
        # Boolean masks
        mask = tf.greater(y_pred, self.mask_value + 1)[:, :, 0]
        mask = tf.cast(mask, tf.float32)
        # Length of each lc
        N = tf.reduce_sum(mask, axis=1)
        # Total number of timesteps
        reps = tf.shape(mask)[1]
        # Repeat the true label $reps times
        y_true = tf.expand_dims(y_true, 1)
        y_true = tf.repeat(y_true, [reps], axis=1)

        original = math_ops.cast(
            math_ops.equal(
                math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
            backend.floatx()
        )

        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(mask)[0], N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1] - N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Substract the first steps to the real mask
        mask = mask - m1

        # Apply the mask
        masked = math_ops.multiply(original, mask)
        # Compute the sum of accuracy along timesteps
        values = tf.reduce_sum(masked, axis=1)
        # Divide the accuracy by each sequence length - N_skip
        values = tf.divide(values, N - N_skip)
        N_batch = tf.cast(tf.shape(mask)[0], tf.float32)
        return tf.reduce_sum(values), N_batch


class CustomTopKAccuracy(tf.keras.metrics.Metric):
    """
    Class that implements the TopK accuracy on the sequence of predictions, padded with a constant value.
    """
    def __init__(self,
                 num_classes:int,
                 k:int=2,
                 N_skip:int=5,
                 mask_value:float=-99.9,
                 **kwargs):
        super(CustomTopKAccuracy, self).__init__(**kwargs)
        """
        Class to compute the topK accuracy per timestep.
        Inputs:
        k: Integer to define the top K. Defaults to 2.
        N_skip: Integer that defines the number of skipped initial timesteps, since the accuracy there is not relevant. Defaults to 5.
        mask_value: Float that defines the value of the mask, to ignore those timesteps. Defaults to -99.9.

        """
        self.num_classes = num_classes
        self.k = k
        self.N_skip = N_skip
        self.mask_value = mask_value
        
        # Compute the signature and apply it using tf.function to self.compute_topk to avoid retracing
        self.compute_signature()
        self.compute_topk = tf.function(func=self.compute_topk,
                                        input_signature=self.signature
                                        )
        self.N = self.add_weight("N_batch", shape=(), initializer="zeros", dtype=tf.float32)
        self.topk = self.add_weight("Batch_TopK", shape=(), initializer="zeros", dtype=tf.float32)
        self.total_topk = self.add_weight("TopK", shape=(), initializer="zeros", dtype=tf.float32)
        

    def compute_signature(self)->None:
        """Define the input signature for the compute_topk method"""
        self.signature = (tf.TensorSpec(shape=[None, self.num_classes], dtype=tf.int32),
                            tf.TensorSpec(shape=[None, None, self.num_classes], dtype=tf.float32))

    def reset_state(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, y_true, y_pred, sample_weight=None):
        topk, N = self.compute_topk(y_true, y_pred)
        self.N.assign_add(N)
        self.topk.assign_add(topk)

        self.total_topk = self.topk / self.N
        return self.total_topk

    def result(self):
        return self.total_topk

    def compute_topk(self, y_true, y_pred):
        """Main function that computes the Mean TopK Accuracy."""
        
        # Boolean mask
        mask = tf.greater(y_pred, self.mask_value + 1)[:, :, 0]
        mask = tf.cast(mask, tf.float32)
        # Length of each lc
        N = tf.reduce_sum(mask, axis=1)
        # Total number of timesteps
        reps = tf.shape(mask)[1]
        # Repeat the true label $reps times
        y_true = tf.expand_dims(y_true, 1)
        y_true = tf.repeat(y_true, [reps], axis=1)

        N_batch = tf.shape(y_pred)[0]
        num_classes = tf.shape(y_pred)[2]
        exp_pred = tf.reshape(y_pred, shape=(N_batch * reps, num_classes))
        exp_true = tf.reshape(y_true, shape=(N_batch * reps, num_classes))
        true = tf.cast(math_ops.argmax(exp_true, axis=-1), tf.int32)
        pred = exp_pred

        res = nn.in_top_k(pred, true, self.k)
        res = math_ops.cast(res, backend.floatx())
        res = tf.reshape(res, (N_batch, reps))

        # All 1 tensor (the ones we want to skip)
        m11 = tf.ones(shape=(tf.shape(mask)[0], self.N_skip), dtype=tf.float32)
        # All ones,( the padding) Note the shape
        m12 = tf.zeros(shape=(tf.shape(mask)[0], tf.shape(mask)[1] - self.N_skip), dtype=tf.float32)
        # Concat both tensors along the time dimension
        m1 = tf.concat((m11, m12), axis=1)
        # Substract the first steps to the real mask
        mask = mask - m1

        # # Apply the mask
        masked = math_ops.multiply(res, mask)
        # # Compute the sum of accuracy along timesteps
        values = tf.reduce_sum(masked, axis=1)
        # # Divide the accuracy by each sequence length - N_skip
        values = tf.divide(values, N - self.N_skip)
        N_batch = tf.cast(N_batch, tf.float32)
        return tf.reduce_sum(values), N_batch
