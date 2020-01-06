import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
import os
from json import dump
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import shift
from numpy.random import shuffle
import Multiband_utils
from tqdm import tqdm

class Prep_Data():
    '''Class that implements functions prepare, read,
    transform and save the data'''
    def __init__(self, version, max_L= 40000, min_L= 500, min_N= 10,
                 w=2, s=1, njobs=7, n_bands=2, max_N = None):

        if type(version) != str:
            version = str(version)
        self.version = version

        # Impose number of min and max light curves per class
        self.max_L = max_L
        self.min_L = min_L

        # Impose a minimum of points per light curve per band
        self.min_N = min_N
        self.max_N = max_N

        # Container for the data
        self.labels = []
        self.matrices = []
        self.orders = []
        self.IDs = []
        self.Freqs = []

        # Auxiliary functions
        self.__func_read = Multiband_utils.read
        self.__func_process = Multiband_utils.process_lc
        self.__func_serialize = Multiband_utils.serialize

        # Parameters for the objects
        self.w = w
        self.s = s
        self.njobs = njobs
        self.n_bands = n_bands

    def set_execution_variables(self, file_train, save_dir, train_size, val_size, test_size
                            , info=None, lc_parameters = None):

        '''Defines paths and split information.
        This function separates the object itself with the different
        excecutions of the object.'''

        # Addresses to store the model and related info
        # self.root_directory = root_directory
        self.lc_parameters = lc_parameters
        # Check and create folders
        self.create_folders()
        # Set inference mode
        # self.inference = inference
        # self.inference_folder = inference_folder
        self.save_dir =  './'+self.version+'/Saved_Data/'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Select the information to include in each input row
        if info is None:
            self.info = [True]*3
        else:
            self.info = info


        self.default_split = True
        # Dataset info
        self.file_train = file_train

        # Splits fractions
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size

        # Set the train/test/val sizes
        # If train_size are files, with the same format as
        #"file_train", use these to create the splits.
        if( type(train_size)==str and type(test_size)==str and type(val_size)==str):
            self.default_split = False
            self.file_train = train_size
            self.file_test = test_size
            self.file_val = val_size

        # Read the datasets to be used. Just one if default splits, three if custom splits.
        self.read_datasets()

        # Extract classes and the number of them
        self.classes = list(set(self.data_train.Class))
        self.num_classes = len(self.classes)

        # Dictionary to transform string labels to ints
        self.trans = {c: n for c, n in zip(self.classes, range(self.num_classes))}
        self.trans_inv = dict(zip(self.trans.values(), self.trans.keys()))

    def read_datasets(self):
        '''Read the dataset, extract the LCs information, with the class and ID.
        Filter the specified number of LC per class, so it does not read everything.

        If default_split is False, each one of the folds is read separately.
        No further filtering is done.'''

        if self.default_split:
            # Read stars Datamenos mal que
            # !!!!!!!!!!!!!!!!!!!!!!!!!MODIFY
            self.data_train = pd.read_csv(self.file_train, usecols=['Address','sourceID','N','N_b','N_r','Class','Class_score','frequency'])

            # Extract classes and the number of them
            self.classes = list(set(self.data_train.Class))
            self.num_classes = len(self.classes)

            # Filter train according to number of observations and elements per class
            self.filter_train()
        else:
            self.data_train= pd.read_csv(self.file_train, usecols=['ID', 'Address','Class','N'])
            self.data_test = pd.read_csv(self.file_test, usecols=['ID', 'Address','Class','N'])
            self.data_val = pd.read_csv(self.file_val, usecols=['ID', 'Address','Class','N'])

    def filter_train(self):
        '''Filter the objects to be read.
        First by imposing restriction to the number of data points.
        Second, by extracting a random sample of up uo max_L elements
        per category.'''
        # Objects that fulfill the number of datapoints condition
        bol1 = self.data_train.N_b>=self.min_N
        bol2 = self.data_train.N_r>=self.min_N
        bol = np.logical_and(bol1,bol2)
        self.data_train = self.data_train[bol]

        # Leave up_to N_max objects per class
        dfs = []
        for i in self.classes.copy():
            # Objects of the class
            bol = self.data_train.Class == i
            sel = self.data_train[bol]

            # Limit the minimum number of light curves
            if sel.shape[0] < self.min_L:
                # Update the classes
                self.classes.remove(i)
                self.num_classes = len(self.classes)
                # Skip the class
                continue

            # Random sample of objects, not done in inference
            # if not self.inference:
            # Return the min among the number of objects and max_L
            num = min(self.max_L, sel.shape[0])
            # Get a random sample
            sel = sel.sample(num, replace=False, axis=0)
            dfs.append(sel)
        # Join the dataframes of each class together
        self.data_train = pd.concat(dfs)

    def create_folders(self):
        # Files to store the data
        # If I publish this, it should accept any address
        self.files_save = self.version+'/'
        # Folder to store TFRecords files
        # self.data_save = self.files_save+'records/'
        # Folder to store saved models
        # self.models_save = self.files_save+'models/'
        if not os.path.exists(self.files_save):
            os.makedirs(self.files_save)
            # os.makedirs(self.data_save)
            # Folders to save the results.
            # os.makedirs(self.files_save+'Results/Data/')
            # os.makedirs(self.files_save+'Results/CMs/')

    def __parallel_read_util(self, _data_):
        '''Reads un parallel light curves in _data_.'''
        iter = zip(_data_.Address, _data_.Class, _data_.sourceID, _data_.frequency)
        ext = Parallel(self.njobs)(delayed(self.__func_read)(address_, class_, id_, f_, self.lc_parameters) for address_, class_, id_, f_ in
                                 tqdm(iter))
        return ext

    def __sort_lcs_util(self, read_lcs):
        '''Create a dictionary, where each class is the key, the id and
        light curve itself are stored in a list, as values.'''

        # Create a dictionary by class
        lcs = {c: [] for c in self.classes}
        # For each class, light curve i[0] and the id i[2]
        [lcs[i[1]].append([i[0], i[2], i[3]]) for i in read_lcs]
        return lcs

    def parallel_read(self):
        '''Run parallel read using n_jobs threads, depending on the user choice.'''
        if self.default_split:
            self.parallel_read_default()
        else:
            self.parallel_read_custom()

    def parallel_read_default(self):
        '''Read the data using n_jobs. Store them in a dict where the classes
        are keys.'''
        # Make the selection here, to avoid reading unnecessary data
        read_lcs = self.__parallel_read_util(self.data_train)
        # Creates the container dictionary, key subclass, value, all the light curves
        self.lcs = self.__sort_lcs_util(read_lcs)

        def parallel_read_custom(self):
            pass

    def parallel_process(self):
        '''Extracts the data and transform it into matrix representation.'''
        if self.default_split:
            self.parallel_process_default()
        else:
            self.parallel_process_custom()

    def parallel_process_custom(self):
        pass

    def __process_lcs_util(self, lcs):
        '''Fucntion to process the lcs given lcs.'''
        Labels = []
        Matrices = []
        IDs = []
        Orders = []
        Freqs = []
        for c in self.classes:
            sel = lcs[c]
            # Run the process function in parallel
            processed = Parallel(self.njobs)(delayed(self.__func_process)(c, l, self.w, self.s, self.n_bands, self.info) for l in tqdm(sel))

            _Labels, _Orders, _Matrices, _IDs, _Freqs = list(zip(*processed))
            # Store in list the information.
            # The order is preserved, so an jth element in all lists will correspond to the same object
            # Change the class to a number, and store it into a list
            _Labels = [self.trans[i] for i in _Labels]

            Labels.append(_Labels)
            Matrices.append(_Matrices)
            IDs.append(_IDs)
            Orders.append(_Orders)
            Freqs.append(_Freqs)

        Labels = np.concatenate(Labels, axis=0)
        Matrices = np.concatenate(Matrices, axis=0)
        IDs = np.concatenate(IDs, axis=0)
        Orders = np.concatenate(Orders, axis=0)
        Freqs = np.concatenate(Freqs, axis=0)

        return Labels, Matrices, IDs, Orders, Freqs

    def parallel_process_default(self):
        '''Extracts the data and transform it into matrix representation.'''

        self.Labels, self.Matrices, self.IDs, self.Orders, self.Freqs = self.__process_lcs_util(self.lcs)

        # Shuffle the data
        lists = [self.Labels, self.Matrices, self.IDs, self.Orders, self.Freqs]
        self.Labels, self.Matrices, self.IDs, self.Orders, self.Freqs = self.__process_shuffle_util(lists)

    def __process_shuffle_util(self, lists):
        '''Shuffles the data.'''
        ind = np.arange(lists[0].shape[0])
        shuffle(ind)
        for i in range(len(lists)):
            lists[i] = lists[i][ind]
        return lists

    def indices(self, train_ids, val_ids, test_ids):

        ind = range(len(self.IDs))
        ind_dict = dict(zip(self.IDs, ind))


        ind_train = list(map(ind_dict.get, train_ids))
        ind_val = list(map(ind_dict.get, val_ids))
        ind_test = list(map(ind_dict.get, test_ids))

        return ind_train, ind_test, ind_val

    def split_train(self):
        self.data_train= self.data_train.set_index('sourceID')

        train_ids, test_val_ids = train_test_split(
            self.data_train.index.values,
            train_size=self.train_size,
            stratify=self.data_train.Class)

        test_ids, val_ids = train_test_split(
            test_val_ids,
            train_size=self.test_size/(1-self.train_size),
            stratify=self.data_train.loc[test_val_ids].Class)

        ind_train, ind_test, ind_val = self.indices(train_ids, val_ids, test_ids)

        self.Matrices_train = self.Matrices[ind_train]
        self.Orders_train = self.Orders[ind_train]
        self.Freqs_train = self.Freqs[ind_train]
        self.Labels_train = self.Labels[ind_train]
        self.IDs_train = self.IDs[ind_train]

        self.Matrices_val = self.Matrices[ind_val]
        self.Labels_val = self.Labels[ind_val]
        self.IDs_val = self.IDs[ind_val]
        self.Freqs_val = self.Freqs[ind_val]
        self.Orders_val = self.Orders[ind_val]

        self.Matrices_test = self.Matrices[ind_test]
        self.Labels_test = self.Labels[ind_test]
        self.IDs_test = self.IDs[ind_test]
        self.Orders_test = self.Orders[ind_test]
        self.Freqs_test = self.Freqs[ind_test]


    def serialize_all(self):
        '''Serialize the data into TFRecords.'''

        self.serialize(self.n_bands, self.Matrices_train,
                       self.Labels_train,
                       self.IDs_train,
                       self.Orders_train,
                       self.Freqs_train,
                       self.save_dir+'Train.tfrecord')

        self.serialize(self.n_bands, self.Matrices_val,
                       self.Labels_val,
                       self.IDs_val,
                       self.Orders_val,
                       self.Freqs_val,
                       self.save_dir+'Val.tfrecord')

        self.serialize(self.n_bands, self.Matrices_test,
                       self.Labels_test,
                       self.IDs_test,
                       self.Orders_test,
                       self.Freqs_test,
                       self.save_dir+'Test.tfrecord')

    def serialize(self, n_bands, Matrices, Labels, IDs, Orders, Freqs, save_path):
        '''Serialize objects given the data and path.'''
        with open(save_path, 'w') as f:
            writer = tf.io.TFRecordWriter(f.name)
            for i in range(len(IDs)):
                ex = self.__func_serialize(n_bands, Matrices[i], Labels[i], IDs[i], Orders[i], Freqs[i])
                writer.write(ex.SerializeToString())

    def write_metadata_process(self):
        '''Write metadata into a file.'''
        self.metadata = {}
        self.metadata['Version'] = self.version
        self.metadata['w'] = self.w
        self.metadata['s'] = self.s
        self.metadata['Max per class'] = self.max_L
        self.metadata['Min per class'] = self.min_L
        self.metadata['Max points per lc'] = self.max_N
        self.metadata['Min points per lc'] = self.min_N
        self.metadata['Number of classes'] = self.num_classes
        self.metadata['Train fraction'] = self.train_size
        self.metadata['Test fraction'] = self.test_size
        self.metadata['Val fraction'] = self.val_size
        self.metadata['Classes Info'] = self.splits_metadata
        self.metadata['Number of bands'] = self.n_bands
        self.metadata['info'] = self.info

        path = self.save_dir+'metadata_preprocess.json'
        with open(path, 'w') as fp:
            dump(self.metadata, fp)
        # Save the light curve parameters for the pandas call
        np.savez(self.save_dir+'lc_parameters',lc_parameters = self.lc_parameters)

    def cls_metadata(self, labels):
        keys, values = np.unique(labels, return_counts= True)
        values_norm = values/sum(values)
        values_norm = ['({0:.3f} %)'.format(100*v) for v in values_norm]
        values = [str(v1)+' '+v2 for v1,v2 in zip(values, values_norm)]
        keys = [self.trans_inv[k] for k in keys]
        hist = dict(zip(keys, values))
        hist['Total'] = len(labels)
        return hist

    def get_metadata_split(self):
        '''Get the metadata of each splits.'''

        splits_labels = [self.Labels_train, self.Labels_test, self.Labels_val]

        values = [self.cls_metadata(labels) for labels in splits_labels]
        keys = ['Train set', 'Test set', 'Val set']
        metadata = dict(zip(keys, values))
        metadata['Keys'] = self.trans_inv
        self.splits_metadata = metadata

    def prepare(self, file_train, save_dir
                , train_size = 0.70, val_size=0.10, test_size=0.2, info=[True, True, True], lc_parameters = None):
                # , inference= False, inference_folder = None):
        self.set_execution_variables( file_train, save_dir
                        , train_size, val_size, test_size, info, lc_parameters)
                        # , inference, inference_folder)
        self.parallel_read()
        self.parallel_process()
        if self.default_split:
            self.split_train()
        self.get_metadata_split()
        self.serialize_all()
        self.write_metadata_process()
