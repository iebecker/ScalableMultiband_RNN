import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
import tensorflow as tf


def read(df, cols, lc_params, n_bands):
    """Function to read the files using all cores

    df: A dataframe of one example
    cols: Columns to be read, [Path, Class, ID] + Param_Phys + Param_Phys_est
    lc_params: dict containing the format of the light curve file
    n_bands: The number of bands of the light curve"""
    # dir_, cls_, ID, t_eff, lum,
    # Read the dataframe for an object, discard the errors
    base = cols[0]
    phys_params = cols[1]
    phys_params_est = cols[2]

    path = df[base[0]]
    cls = df[base[1]]
    ID = df[base[2]]

    lc = pd.read_csv(path, **lc_params)
    lc = lc.dropna(how='any').values
    # The default column order is
    # filter, time, mag, magerr, order

    # Split bands
    # Here I have to implement a translator, between name in the dataset file,
    # and name in the light curve file.
    # For now, incrementing from 1.
    lc_bands = []
    for band in range(n_bands):
        b = lc[:, 0] == band + 1
        # time, mag, mag_err, order
        lc_band = lc[b][:, [1, 2, 3, 4]]
        lc_bands.append(lc_band)

    # Extract physical info if required

    phys_params_out = {}
    if len(phys_params) > 0:
        # phys_params = [df[i] for i in phys_params]
        phys_params_out = {i: df[i] for i in phys_params}

    # Extract physical estimates info if required.
    # Specifically for T_eff est
    phys_params_est_out = {}
    if len(phys_params_est) > 0:
        phys_params_est_out = {i: df[i] for i in phys_params_est}
    return lc_bands, cls, ID, phys_params_out, phys_params_est_out


def process_lc(cls_,
               _data_,
               w,
               s,
               n_bands,
               max_N=np.inf,
               ):
    """Process one single element, to be excecuted in parallel.
    "matrices" must be and object containing all the information used for input.
    New in version 2"""
    # Info contain which info to include. 0 only mag, 1 time and mag and

    data_ = _data_[0]
    ID = _data_[1]

    phys_params = _data_[2]
    phys_params_est = _data_[3]

    # Process physical parameters
    # if -1, means NaN. Compute the Log if greater than 0, else, 0
    phys_values = {}
    for key in phys_params.keys():
        phys_values[key] = phys_params[key] if phys_params[key] > 0 else 0

    phys_values_est = {}
    for key in phys_params_est.keys():
        phys_values_est[key] = (phys_params_est[key] - 3000) / 1.0e3 if phys_params_est[key] > 0 else 0

    # Columns:
    # 0: time
    # 1: mag
    # 2: mag_err
    # 3: order

    # Order, light curve and DT information containers
    matrices = [[]] * n_bands
    orders = [[]] * n_bands
    uncertainty = [[]] * n_bands

    T0 = [[]] * n_bands  # initial time t_0
    M0 = [[]] * n_bands  # magnitude at t_0

    # Substract the initial time minus one
    # Find all the starting times
    min_t = np.min([np.min(data_[b][:, 0]) for b in range(n_bands)])

    # # All the lcs start at time 1, so the log10 is 0
    # min_t-=1
    # print(ID)
    for b in range(n_bands):
        # Cut to a max_N values per band
        data_[b] = data_[b][:max_N + 2, :]

        # Extract time and mag to compute deltas
        sel = data_[b][:, [0, 1]]

        # # Substract the min_t
        sel[:, 0] = sel[:, 0] - min_t

        # Compute the differences
        d = sel - shift(sel, [1, 0], cval=0)

        # Remove the first measument because it is unchanged
        d = d[1:]

        # Compute the log10 of the delta times
        d[:, 0] = np.log10(d[:, 0])

        # Extract first time t0
        T0[b] = sel[0, 0]

        # Extract mag at t0
        M0[b] = sel[0, 1]

        # Place the data in each window
        N = int((d.shape[0] - w + s) / s)
        D_Time = [d[i * s:i * s + w, 0] for i in range(N)]
        D_Mag = [d[i * s:i * s + w, 1] for i in range(N)]

        # Create the input matrix in band b
        matrices[b] = np.concatenate((D_Time, D_Mag), axis=1).astype(np.float32)
        # Order information, discard first w elements
        orders[b] = data_[b][w:, 3].astype(np.int32)

        # Construct the uncertainty for each band
        # Skip the first observation of each band.
        uncertainty[b] = data_[b][1:][w - 1::s, 2].astype(np.float32)

    # Compute the minimum order and substract it
    # So all of them start with 0
    min_order = np.min([np.min(orders[b]) for b in range(n_bands)])
    orders = [orders[b] - min_order for b in range(n_bands)]

    out_dict = {'Label': cls_, 'Order': orders, 'ID': ID, 'Physical_Values': phys_values,
                'Estimated_Physical_Values': phys_values_est, 'Matrices': matrices, 'M_0': M0, 'T_0': T0,
                'Uncertainty': uncertainty}
    return out_dict


def serialize(dict_,
              ):
    # Get the number of bands
    n_bands = len(dict_['M_0'])

    dict_features = {
        'ID': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(dict_['ID']).encode()])),
        'Label': tf.train.Feature(int64_list=tf.train.Int64List(value=[dict_['Label']])),
    }

    # Add the physical values
    for key in dict_['Physical_Values'].keys():
        dict_features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[dict_['Physical_Values'][key]]))
    # Add the estimation of the physical valeus
    for key in dict_['Estimated_Physical_Values'].keys():
        dict_features[key] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[dict_['Estimated_Physical_Values'][key]]))

    # Add contextual information per band
    for i in range(n_bands):
        # Length of each band
        dict_features["N_" + str(i)] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[dict_['Matrices'][i].shape[0]]))
        # First magnitude
        dict_features["M0_" + str(i)] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[dict_['M_0'][i]]))
        # First time
        dict_features["T0_" + str(i)] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[dict_['T_0'][i]]))

    element_context = tf.train.Features(feature=dict_features)

    # Feature lists the sequential feature of our example
    dict_sequence = {}
    for i in range(n_bands):
        lc_ = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=dict_['Matrices'][i].ravel()))
        o_ = tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=dict_['Order'][i].ravel()))
        u_ = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=dict_['Uncertainty'][i].ravel()))

        lcs = tf.train.FeatureList(feature=[lc_])
        os = tf.train.FeatureList(feature=[o_])
        us = tf.train.FeatureList(feature=[u_])

        dict_sequence['LightCurve_' + str(i)] = lcs
        dict_sequence['Order_' + str(i)] = os
        dict_sequence['Uncertainty_' + str(i)] = us

    element_lists = tf.train.FeatureLists(feature_list=dict_sequence)

    # The object we return
    example = tf.train.SequenceExample(context=element_context,
                                       feature_lists=element_lists
                                       )

    return example
