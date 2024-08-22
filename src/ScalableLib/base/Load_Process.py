import os
import pickle
from json import dump

import ScalableLib.base.Utils as utils
import numpy as np
import pandas as pd
import tensorflow as tf
from ScalableLib.classifier.CustomScalers import ParamPhysScaler
from joblib import Parallel, delayed
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class prepData:
    """Class that implements functions prepare, read, transform and save the data"""

    def __init__(self,
                 max_l=40000,
                 min_l=500,
                 min_n=10,
                 w=2,
                 s=1,
                 njobs=7,
                 n_bands=2,
                 max_n=None,
                 ):

        # Impose number of min and max light curves per class
        self.data_val = None
        self.classes = None
        self.dataset_header = None
        self.save_dir = None
        self.lc_parameters = None
        self.num_classes = None
        self.max_l = max_l
        self.min_l = min_l

        # Impose a minimum of points per light curve per band
        self.min_n = min_n
        self.max_n = max_n

        # Container for the data
        self.labels = []
        self.matrices = []
        self.orders = []
        self.IDs = []

        # Auxiliary functions
        self.__func_read = utils.read
        self.__func_process = utils.process_lc
        self.__func_serialize = utils.serialize
        # Parameters for the objects
        self.w = w
        self.s = s
        self.njobs = njobs
        self.n_bands = n_bands

        self.mask_value = -99.99

    def set_execution_variables(self,
                                file_train,
                                file_val,
                                file_test,
                                save_dir,
                                dataset_header,
                                train_size,
                                val_size,
                                test_size,
                                lc_parameters,
                                params_phys,
                                params_phys_est,
                                elements_per_shard,
                                ):

        """Defines paths and split information.
        This function separates the object itself with the different
        executions of the object."""

        # Addresses to store the model and related info
        self.lc_parameters = lc_parameters

        self.dataset_header = dataset_header

        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Define the type of split
        self.default_split = True
        self.custom_test_split = False
        self.custom_splits = False
        # Dataset info
        self.file_train = file_train
        self.file_test = file_test
        self.file_val = file_val

        # Regression files
        self.params_phys = params_phys
        self.params_phys_est = params_phys_est
        # Splits fractions
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        # Set the train/test/val sizes
        # If train_size are files, with the same format as
        # "file_train", use these to create the splits.
        if self.file_val is not None and self.file_test is not None:
            self.default_split = False
            self.custom_splits = True

        elif self.file_val is None and self.file_test is not None:
            self.default_split = False
            self.custom_test_split = True

        # Read the datasets to be used. Just one if default splits, three if custom splits.
        self.read_datasets()

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)

        # Dictionary to transform string labels to ints
        self.trans = {c: n for c, n in zip(self.classes, range(self.num_classes))}
        self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))
        self.elements_per_shard = elements_per_shard

    def read_datasets(self):
        """Read the dataset, extract the LCs information, with the class and ID.
        Filter the specified number of LC per class, so it does not read everything.

        If default_split is False, each one of the folds is read separately.
        No further filtering is done."""
        # Read stars Data
        self.cols = self.dataset_header + self.params_phys

        if self.default_split:
            self.data_train = pd.read_csv(self.file_train,
                                          usecols=self.cols)
        elif self.custom_test_split:
            self.data_train = pd.read_csv(self.file_train, usecols=self.cols)
            self.data_test = pd.read_csv(self.file_test, usecols=self.cols)
        elif self.custom_splits:
            self.data_train = pd.read_csv(self.file_train, usecols=self.cols)
            self.data_test = pd.read_csv(self.file_test, usecols=self.cols)
            self.data_val = pd.read_csv(self.file_val, usecols=self.cols)

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)
        # Find the band names
        self.band_names = [i[2:] for i in self.data_train.columns if 'N_' in i]

        # Filter train according to number of observations and elements per class
        if self.default_split:
            self.data_train = self.filter(self.data_train)
        elif self.custom_test_split:
            self.data_train = self.filter(self.data_train)
            self.data_test = self.filter(self.data_test)
        elif self.custom_splits:
            # The data should come filtered beforehand
            pass

    def filter(self, data):
        """Filter the objects to be read.
        First by imposing restriction to the number of data points.
        Second, by extracting a random sample of up to max_l elements
        per category."""

        # Objects that fulfill the number of datapoints condition
        bols = np.ones((data.shape[0], len(self.band_names)), dtype=np.bool)
        for i in range(len(self.band_names)):
            bols[:, i] = data['N_' + self.band_names[i]] > self.min_n
        bol = bols.sum(axis=1) == len(self.band_names)
        data = data[bol]

        # Leave up_to N_max objects per class
        dfs = []
        for i in self.classes.copy():
            # Objects of the class
            bol = data.Class == i
            sel = data[bol]

            # Limit the minimum number of light curves
            if sel.shape[0] < self.min_l:
                # Update the classes
                self.classes.remove(i)
                self.num_classes = len(self.classes)
                # Skip the class
                continue

            # Return the min among the number of objects and max_l
            num = min(self.max_l, sel.shape[0])
            # Get a random sample
            sel = sel.sample(num, replace=False, axis=0)
            dfs.append(sel)
        # Join the dataframes of each class together
        data = pd.concat(dfs)
        return data

    def __parallel_read_util(self, _data_):
        """Reads un parallel light curves in _data_."""

        read_cols = ['Path',
                     'Class',
                     'ID'
                     ]
        read_cols = [read_cols, self.params_phys, self.params_phys_est]
        # Filter;Time;Mag;Mag_err;Order
        ext = Parallel(self.njobs)(delayed(self.__func_read)(_data_.iloc[i],
                                                             read_cols,
                                                             self.lc_parameters,
                                                             self.n_bands,
                                                             ) for i in tqdm(range(_data_.shape[0])))
        return ext

    def __sort_lcs_util(self, read_lcs):
        """Create a dictionary, where each class is the key, the id and
        light curve itself are stored in a list, as values."""

        # Create a dictionary by class
        lcs = {c: [] for c in self.classes}
        # For each class,
        # i[0] light curve data
        # i[2] id
        # i[3] phys params
        # i[4] phys_params_est
        [lcs[i[1]].append([i[0], i[2], i[3], i[4]]) for i in read_lcs]
        return lcs

    def parallel_read(self):
        """Run parallel read using n_jobs threads, depending on the user choice."""
        if self.default_split:
            self.parallel_read_default()
        else:
            self.parallel_read_custom()

    def parallel_read_default(self):
        """Read the data using n_jobs. Store them in a dict_transform where the classes
        are keys."""
        # Make the selection here, to avoid reading unnecessary data
        self.read_lcs = self.__parallel_read_util(self.data_train)
        # Creates the container dictionary, key subclass, value, all the light curves
        self.lcs = self.__sort_lcs_util(self.read_lcs)

    def parallel_read_custom(self):
        """Read all the datasets using n_jobs. Store each split in a dict_transform where the classes
        are keys."""

        # Process according to the different experimental setup
        if self.custom_test_split:
            # Make the selection here, to avoid reading unnecessary data
            self.read_lcs_train = self.__parallel_read_util(self.data_train)
            self.read_lcs_test = self.__parallel_read_util(self.data_test)

            # Creates the container dictionary, key subclass, value, all the light curves
            self.lcs_train = self.__sort_lcs_util(self.read_lcs_train)
            self.lcs_test = self.__sort_lcs_util(self.read_lcs_test)

        elif self.custom_splits:
            # Make the selection here, to avoid reading unnecessary data
            self.read_lcs_train = self.__parallel_read_util(self.data_train)
            self.read_lcs_val = self.__parallel_read_util(self.data_val)
            self.read_lcs_test = self.__parallel_read_util(self.data_test)

            # Creates the container dictionary, key subclass, value, all the light curves
            self.lcs_train = self.__sort_lcs_util(self.read_lcs_train)
            self.lcs_val = self.__sort_lcs_util(self.read_lcs_val)
            self.lcs_test = self.__sort_lcs_util(self.read_lcs_test)

    def parallel_process(self):
        """Extracts the data and transform it into matrix representation."""
        if self.default_split:
            self.parallel_process_default()
        else:
            self.parallel_process_custom()

    def parallel_process_custom(self):
        """Extracts the data from each split and transform it into matrix representation.
        """
        if self.custom_test_split:
            out_dict_train = self.__process_lcs_util(self.lcs_train)
            out_dict_test = self.__process_lcs_util(self.lcs_test)

            self.dict_train = self.__process_shuffle_util(out_dict_train)
            self.dict_test = self.__process_shuffle_util(out_dict_test)

        elif self.custom_splits:
            out_dict_train = self.__process_lcs_util(self.lcs_train)
            out_dict_val = self.__process_lcs_util(self.lcs_val)
            out_dict_test = self.__process_lcs_util(self.lcs_test)

            self.dict_train = self.__process_shuffle_util(out_dict_train)
            self.dict_val = self.__process_shuffle_util(out_dict_val)
            self.dict_test = self.__process_shuffle_util(out_dict_test)

    def __process_lcs_util(self, lcs):
        """Function to process the lcs."""

        all_processed = []
        for c in self.classes:
            sel = lcs[c]
            # Run the process function in parallel
            # Returns a list of dicts
            processed = Parallel(self.njobs)(delayed(self.__func_process)(c,
                                                                          l,
                                                                          self.w,
                                                                          self.s,
                                                                          self.n_bands,
                                                                          self.max_n,
                                                                          ) for l in tqdm(sel))

            all_processed.append(processed)

        # Concatenate the elements
        all_processed = [j for i in all_processed for j in i]

        # Create the container
        keys = all_processed[0].keys()  # Get the keys from the first element
        output_dict = {}
        for key in keys:  # From dicts copies the same []
            output_dict[key] = []
        for elem in all_processed:
            for key in keys:
                output_dict[key].append(elem[key])
        # Transform lists to np arrays
        for key in keys:
            if key == 'Label':  # Transform labels to numbers
                output_dict[key] = [self.trans[i] for i in output_dict[key]]
            output_dict[key] = np.array(output_dict[key], dtype='object')

        return output_dict

    def parallel_process_default(self):
        """Extracts the data and transform it into matrix representation.
        New in version 2"""

        out_dict = self.__process_lcs_util(self.lcs)

        self.shuffled_dict = self.__process_shuffle_util(out_dict)

    @staticmethod
    def __process_shuffle_util(dict):
        """Shuffles the data."""
        # Get the keys
        keys = list(dict.keys())
        # Get integer from 0 to the number of elements-1
        ind = np.arange(dict[keys[0]].shape[0])
        # Shuffle the integers
        shuffle(ind)
        # Shuffle each element of the dict_transform
        for i in keys:
            dict[i] = dict[i][ind]
        return dict

    def indices_custom_test(self, train_ids, val_ids):
        ind = range(len(self.dict_train['ID']))
        ind_dict = dict(zip(self.dict_train['ID'], ind))

        ind_train = list(map(ind_dict.get, train_ids))
        ind_val = list(map(ind_dict.get, val_ids))
        return ind_train, ind_val

    def indices_default(self, train_ids, val_ids, test_ids):

        ind = range(len(self.shuffled_dict['ID']))
        ind_dict = dict(zip(self.shuffled_dict['ID'], ind))

        ind_train = list(map(ind_dict.get, train_ids))
        ind_val = list(map(ind_dict.get, val_ids))
        ind_test = list(map(ind_dict.get, test_ids))

        return ind_train, ind_test, ind_val

    def fit_scalers(self):
        """Fit and save the scalers (0.1,1.1) for each phys param"""

        self.scalers = {}
        df = pd.DataFrame(list(self.dict_train['Physical_Values']))
        for var in self.params_phys:
            nonzero = df[var]
            b = nonzero > 0
            nonzero = nonzero[b]

            self.scalers[var] = ParamPhysScaler(param=var,
                                                mask_value=self.mask_value,
                                                )
            self.scalers[var].fit(nonzero.values.reshape(-1, 1))

    def scalers_transform(self, dict_transform):
        # Create a DataFrame to transform the data at once
        df = pd.DataFrame(list(dict_transform['Physical_Values']))

        for col in df.columns:
            # Transform the entire column
            df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1),
                                                  ).ravel()

        # Save scalers
        path = self.save_dir + 'scalers.pkl'
        with open(path, 'wb') as fp:
            pickle.dump(self.scalers, fp)

        # Return to the dict_transform representation
        reverse = df.transpose().to_dict()
        dict_transform['Physical_Values'] = np.array([reverse[i] for i in reverse])

        return dict_transform

    def split_custom_test(self):
        self.data_train = self.data_train.set_index('ID')

        train_ids, val_ids = train_test_split(self.data_train.index.values,
                                              train_size=self.train_size,
                                              test_size=self.val_size,
                                              stratify=self.data_train.Class,
                                              shuffle=True)

        ind_train, ind_val = self.indices_custom_test(train_ids, val_ids)

        self.dict_val = {}

        for key in self.dict_train.keys():
            self.dict_val[key] = self.dict_train[key][ind_val]
            self.dict_train[key] = self.dict_train[key][ind_train]

    def split_default(self):
        self.data_train = self.data_train.set_index('ID')

        train_ids, test_val_ids = train_test_split(
            self.data_train.index.values,
            train_size=self.train_size,
            stratify=self.data_train.Class)

        test_ids, val_ids = train_test_split(
            test_val_ids,
            train_size=self.test_size / (1 - self.train_size),
            stratify=self.data_train.loc[test_val_ids].Class)

        ind_train, ind_test, ind_val = self.indices_default(train_ids, val_ids, test_ids)

        self.dict_train = {}
        self.dict_test = {}
        self.dict_val = {}

        for key in self.shuffled_dict.keys():
            self.dict_train[key] = self.shuffled_dict[key][ind_train]
            self.dict_test[key] = self.shuffled_dict[key][ind_test]
            self.dict_val[key] = self.shuffled_dict[key][ind_val]

    def split(self):
        """ Split the data intro train-test-val according to the experimental setup.
        """
        if self.default_split:
            self.split_default()
        elif self.custom_test_split:
            self.split_custom_test()

    def scale_datasets(self):
        # Fit the scalers
        self.fit_scalers()
        # Transform the datasets
        self.dict_train = self.scalers_transform(self.dict_train)
        self.dict_test = self.scalers_transform(self.dict_test)
        self.dict_val = self.scalers_transform(self.dict_val)

    def normalize_phys(self):
        pass

    def shard_serialize_all(self):
        """Serialize the data into TFRecords."""

        self.shard_serialize(self.dict_train,
                             'train',
                             elements_per_shard=self.elements_per_shard)

        self.shard_serialize(self.dict_test,
                             'test',
                             elements_per_shard=self.elements_per_shard)

        self.shard_serialize(self.dict_val,
                             'val',
                             elements_per_shard=self.elements_per_shard)

    def serialize_all(self):
        """Serialize the data into TFRecords."""

        self.serialize(self.dict_train,
                       self.save_dir + 'Train.tfrecord')

        self.serialize(self.dict_test,
                       self.save_dir + 'Test.tfrecord')

        self.serialize(self.dict_val,
                       self.save_dir + 'Val.tfrecord')

    def serialize(self,
                  dict,
                  save_path):
        """Serialize objects given the data and path."""

        keys = dict.keys()

        with open(save_path, 'w') as f:
            writer = tf.io.TFRecordWriter(f.name)
            for i in range(len(dict['ID'])):
                # Get one example in the form of a dict_transform
                temp = {key: dict[key][i] for key in keys}
                ex = self.__func_serialize(temp)
                writer.write(ex.SerializeToString())

    def shard_serialize(self,
                        dict,
                        fold,
                        elements_per_shard=5000,
                        ):
        """Serialize objects given the data and path,
        splitting them into shards."""
        # Create the folders to store the shards
        fold_dir = '/'.join([self.save_dir, fold])
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)

        keys = dict.keys()

        # Number of objects in the split
        N = len(dict['ID'])
        # Compute the number of shards
        n_shards = -np.floor_divide(N, -elements_per_shard)
        # Number of characters of the number of shards
        name_length = len(str(n_shards))

        # Create one file per shard
        for shard in range(n_shards):
            # Get the shard number padded with 0s
            shard_name = str(shard + 1).rjust(name_length, '0')
            # Get the shard store name
            shard_name = '_'.join([fold, shard_name, str(n_shards)])
            # Add the extension
            shard_name = shard_name + '.tfrecord'
            # Get the shard save path
            shard_path = '/'.join([self.save_dir, fold, shard_name])

            with open(shard_path, 'w') as f:
                writer = tf.io.TFRecordWriter(f.name)
                i_ini = shard * elements_per_shard
                i_end = elements_per_shard * (shard + 1)
                for ii in range(i_ini, i_end):
                    if ii > N - 1:
                        break
                    # Get one example in the form of a dict_transform
                    temp = {key: dict[key][ii] for key in keys}
                    # Obtain the serialized example
                    ex = self.__func_serialize(temp)
                    # Write it to a file
                    writer.write(ex.SerializeToString())

    # def shard_serialize_parallel(self,
    #                              dict,
    #                              fold,
    #                              elements_per_shard=5000,
    #                              ):
    #     """Serialize objects given the data and path,
    #     splitting them into shards."""
    #     # Create the folders to store the shards
    #     fold_dir = '/'.join([self.save_dir, fold])
    #     if not os.path.exists(fold_dir):
    #         os.makedirs(fold_dir)

    #     keys = dict.keys()

    #     # Number of objects in the split
    #     N = len(dict['ID'])
    #     # Compute the number of shards
    #     n_shards = -np.floor_divide(N, -elements_per_shard)
    #     # Number of characters of the number of shards
    #     name_length = len(str(n_shards))

    #     # Create one file per shard
    #     shard_paths = []
    #     for shard in range(n_shards):
    #         # Get the shard number padded with 0s
    #         shard_name = str(shard + 1).rjust(name_length, '0')
    #         # Get the shard store name
    #         shard_name = '_'.join([fold, shard_name, str(n_shards)])
    #         # Add the extension
    #         shard_name = shard_name + '.tfrecord'
    #         # Get the shard save path
    #         shard_path = '/'.join([self.save_dir, fold, shard_name])
    #         shard_paths.append(shard_path)

    #     Parallel(self.njobs, backend='threading')(delayed(aux_serialize)(shard,
    #                                                                      shard_path,
    #                                                                      elements_per_shard,
    #                                                                      list(keys),
    #                                                                      N,
    #                                                                      dict)
    #                                               for shard, shard_path in tqdm(enumerate(shard_paths)))

    def write_metadata_process(self):
        """Write metadata into a file."""
        self.metadata = {'w': self.w, 's': self.s, 'Max per class': self.max_l, 'Min per class': self.min_l,
                         'Max points per lc': self.max_n, 'Min points per lc': self.min_n,
                         'Number of classes': self.num_classes, 'Train fraction': self.train_size,
                         'Test fraction': self.test_size, 'Val fraction': self.val_size,
                         'Classes Info': self.splits_metadata, 'Number of bands': self.n_bands,
                         'Physical_parameters': self.params_phys, 'Physical_parameters_est': self.params_phys_est}

        path = self.save_dir + 'metadata_preprocess.json'
        with open(path, 'w') as fp:
            dump(self.metadata, fp)
        # Save the light curve parameters for the pandas call
        np.savez(self.save_dir + 'lc_parameters', lc_parameters=self.lc_parameters)

    def cls_metadata(self, labels):
        keys, values = np.unique(labels, return_counts=True)
        values = [str(v1) for v1 in values]
        keys = [self.trans_inv[k] for k in keys]
        hist = dict(zip(keys, values))

        return hist

    def get_metadata_split(self):
        """Get the metadata of each split."""

        splits_labels = [
            self.dict_train['Label'],
            self.dict_test['Label'],
            self.dict_val['Label']
        ]

        values = [self.cls_metadata(labels) for labels in splits_labels]
        keys = ['Train set', 'Test set', 'Val set']
        metadata = dict(zip(keys, values))
        metadata['Keys'] = self.trans_inv
        self.splits_metadata = metadata

    def prepare(self,
                file_train,
                file_val,
                file_test,
                save_dir,
                dataset_header,
                params_phys=None,
                params_phys_est=None,
                train_size=0.70,
                val_size=0.10,
                test_size=0.2,
                lc_parameters=None,
                elements_per_shard=5000,
                ):

        if params_phys_est is None:
            params_phys_est = []
        if params_phys is None:
            params_phys = []
        self.set_execution_variables(file_train,
                                     file_val,
                                     file_test,
                                     save_dir,
                                     dataset_header,
                                     train_size,
                                     val_size,
                                     test_size,
                                     lc_parameters,
                                     params_phys,
                                     params_phys_est,
                                     elements_per_shard,
                                     )
        self.parallel_read()
        self.parallel_process()
        self.split()
        self.scale_datasets()
        self.get_metadata_split()
        self.shard_serialize_all()
        self.write_metadata_process()
