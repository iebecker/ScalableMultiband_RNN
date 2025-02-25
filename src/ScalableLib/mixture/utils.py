from itertools import combinations
import numpy as np
import tensorflow as tf

class LastRelevantLayer_mod(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LastRelevantLayer_mod, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, output, length):
        '''Get the last relevant output from the network'''
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = tf.shape(output)[2]

        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        
        relevant = tf.reshape(relevant, [batch_size, output.shape[2]])
        return relevant
        
def compute_FinalMeanMags(output, length):

    batch_size = tf.shape(output)[0]
    indices = tf.stack([tf.range(0, batch_size), length], axis=1)
    final_MeanMags = tf.gather_nd(output, indices)
    return final_MeanMags

def compute_ColorMatrix(n_bands):
    ''' Obtains the matrix to compute the colors (magnitude differences)
    between all the bands. In the 2-band scenario it reduces to [[1, -1]].
    '''
    NN = n_bands
    # Compute all possible combinations
    combs = set(combinations(np.arange(NN),2))
    combs = [list(i) for i in combs]
    combs = tf.constant(combs)

    # Compute the index of the first elements
    v1 = tf.reshape(combs[:,0],[-1,1])
    v1b =tf.reshape(tf.range(tf.shape(combs)[0]), [-1,1])
    v1 = tf.concat([v1b, v1], axis=1)
    # Compute the index of the second elements
    v2 = tf.reshape(combs[:,1],[-1,1])
    v1b =tf.reshape(tf.range(tf.shape(combs)[0]), [-1,1])
    v2 = tf.concat([v1b, v2], axis=1)

    # Concatenate and cast to int64
    indices= tf.concat([v1, v2], axis=0)
    indices = tf.cast(indices, tf.int64)

    # First part are the first elements
    u1 = tf.ones(tf.shape(v1)[0])
    # Second part are the subtracted elements
    u2 = -tf.ones(tf.shape(v1)[0])
    updates= tf.concat([u1,u2], axis=0)

    # Compute the shape #combinations x bands
    shape = tf.Variable((tf.shape(v1)[0],NN))
    shape = tf.cast(shape, tf.int64)

    # Obtain the matrix
    diff_matrix = tf.scatter_nd(indices, updates, shape)

    # Transpose it to perform MatMult afterwards
    diff_matrix = tf.transpose(diff_matrix)
    return diff_matrix