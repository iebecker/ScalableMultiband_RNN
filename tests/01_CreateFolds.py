import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import os


def sample_classes(data):
    # Leave up_to N_max objects per class
    dfs = []
    classes = list(data.Class.unique())
    num_classes = len(classes)

    for i in classes.copy():
        # Objects of the class
        bol = data.Class == i
        sel = data[bol]

        # Limit the minimum number of light curves
        if sel.shape[0] < min_l:
            # Update the classes
            classes.remove(i)
            num_classes = len(classes)
            # Skip the class
            continue

        # Return the min among the number of objects and max_l
        num = min(max_l, sel.shape[0])
        # Get a random sample
        sel = sel.sample(num, replace=False, axis=0)
        dfs.append(sel)
    # Join the dataframes of each class together
    data = pd.concat(dfs)
    return data


survey = 'Gaia'

path = '/home/Data/Paper_2/Prepare_dataset/Gaia/V5/Dataset_Gaia_Phys_V5.dat'
# TO DO: Add the dataset to the GitHub
df = pd.read_csv(path)
df.head()

min_n = 10
max_n = 1000

max_l = 10000
min_l = 500

bands = [i for i in df.columns if 'N_' in i]
b = np.ones(df.shape[0], dtype=np.bool_)
for band in bands:
    b_band = df[band] > min_n
    b = np.logical_and(b, b_band)

df = sample_classes(df)
df = df[b].copy()
df = df.reset_index().drop('index', axis=1)

kfolds = StratifiedKFold(n_splits=3, shuffle=True, )
path_folds = './Folds'
if not os.path.exists(path_folds):
    os.mkdir(path_folds)
# First split test
df_temp, df_test = train_test_split(df, stratify=df.Class, train_size=0.8)
df_temp.reset_index(inplace=True)
df_test.reset_index(inplace=True)

path_test = os.path.join(path_folds, 'test.csv')
df_test.to_csv(path_test, index=False, index_label=False)

for n, (train_index, val_index) in enumerate(kfolds.split(df_temp.index.values, df_temp.Class.values)):
    # Get the train and validation splits
    df_train = df_temp.loc[train_index]
    df_val = df_temp.loc[val_index]

    path_folds_ = os.path.join(path_folds, 'Fold_' + str(n + 1))
    if not os.path.exists(path_folds_):
        os.mkdir(path_folds_)

    path_train = os.path.join(path_folds_, 'train.csv')

    path_val = os.path.join(path_folds_, 'val.csv')

    df_train.to_csv(path_train, index=False, index_label=False)
    df_val.to_csv(path_val, index=False, index_label=False)
