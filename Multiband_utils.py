import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import tensorflow as tf


def read(dir_, cls_, ID, freq, lc_params):
    '''Function to read the files using all cores'''

    # Read the dataframe for an object, discard the errors
    df_lc = pd.read_csv(dir_, **lc_params) #header=0, na_filter=True
                        #, sep=',', names=['band', 'time', 'mag', 'Order'])

    # Create the absolute time difference DT
    DT = df_lc.time - shift(df_lc.time, [1], cval=0)
    df_lc = df_lc.assign(DT=DT)

    # Transform into a np.array
    lc = df_lc.values

    # Split bands
    bol_b = lc[:,0] == 'BP'
    bol_r = lc[:,0] == 'RP'

    # Export the light curves
    lc_b = lc[bol_b][:,[1,2,3,4]].astype(np.float32)
    lc_r = lc[bol_r][:,[1,2,3,4]].astype(np.float32)

    return [lc_b, lc_r], cls_, ID, freq

def process_lc(cls_, _data_, w, s, n_bands, info):
    '''Process one single element, to be excecuted in parallel.
    "matrices" must be and object containing all the information used for input.'''
    # Info contain which info to include. 0 only mag, 1 time and mag and
    # 3, time independent of band

    ID = _data_[1]
    freq = _data_[2]
    data_ = _data_[0]
    # Columns:
    # 0: time
    # 1: mag
    # 2: order
    # 3: DT (difference without considering the band)

    # Order, light curve and DT information containers
    matrices = [[]]*n_bands
    orders = [[]]*n_bands
    DT = [[]]*n_bands
    for b in range(n_bands):
        sel = data_[b][:, [0,1]]
        # Make the differences
        d = sel - shift(sel, [1, 0], cval=0)
        # Remove the first measument because it is unchanged
        d = d[1:]

        # Place the data in each window
        N = (d.shape[0]-w+s)/s
        if info[0]:
            D_Time = [d[i*s:i*s+w, 0] for i in range(int(N))]
        if info[1]:
            D_Mag = [d[i*s:i*s+w, 1] for i in range(int(N))]
        if info[2]:
            DT = [data_[b][i*s:i*s+w, 3] for i in range(int(N))]

        # Create the input matrix in band b
        if info[0] and info[1] and not info[2]:
            matrices_b = np.concatenate((D_Time, D_Mag), axis=1)
        elif info[0] and info[1] and info[2]:
            matrices_b = np.concatenate((D_Time, D_Mag, DT), axis=1)

        # Add it to the final object
        matrices[b] = matrices_b.astype(np.float32)

        # Order information, discard first element
        orders[b] = data_[b][w:, 2].astype(np.int32)

    return cls_, orders, matrices, ID, freq

def serialize(n_bands, sequence, label, id, order, frequency):

    dict_features={
    'ID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(id).encode()])),
    'Label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    'Frequency': tf.train.Feature(float_list=tf.train.FloatList(value=[frequency])),
    }

    for i in range(n_bands):
        dict_features["N_"+str(i)] = tf.train.Feature(
        int64_list=tf.train.Int64List(value=[sequence[i].shape[0]]))

    element_context = tf.train.Features(feature = dict_features)

    # Feature lists the sequential feature of our example
    dict_sequence = {}
    for i in range(n_bands):
        lc_ = tf.train.Feature(
                    float_list=tf.train.FloatList(
                                        value=sequence[i].ravel()))
        o_ = tf.train.Feature(
                    int64_list=tf.train.Int64List(
                                        value=order[i].ravel()))

        lcs = tf.train.FeatureList(feature=[lc_])
        os = tf.train.FeatureList(feature=[o_])
        dict_sequence['LightCurve_'+str(i)]= lcs
        dict_sequence['Order_'+str(i)]= os

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)

    # The object we return
    example = tf.train.SequenceExample(
        context= element_context,
        feature_lists= element_lists)

    return example
