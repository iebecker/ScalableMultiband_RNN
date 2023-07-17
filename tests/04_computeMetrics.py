import numpy as np
import sys
import tensorflow as tf
from glob import glob
import os
sys.path.append('../')
sys.path.append('./../src/')

import classifier.Multiband as multiband
import base.plot as plot

# To reset cuda
# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm
# To see if the system recognises the GPU
device = 0
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(devices[device], 'GPU')
tf.config.experimental.set_memory_growth(device=devices[device], enable=True)

survey = 'Gaia'
path = './Results/Fold_*/'
folds = glob(path)
folds.sort()

if not os.path.exists('./Results/'):
    os.mkdir('./Results/')

label_order = ['CEP', 'T2CEP', 'MIRA_SR', 'RRAB', 'RRC', 'DSCT_SXPHE']

# Containers to store the results
reports_folds = []
cm_folds = []
regressions = []
for fold in folds:
    tf.keras.backend.clear_session()
    # Set the fold path
    base_dir = fold + '/'
    # base_dir = fold
    # Get the last run
    path_runs_folder = os.path.join(base_dir, 'Models', '*')
    path_runs = glob(path_runs_folder)
    path_runs.sort()
    path_run = path_runs[-1]

    # path_preprocess = fold.replace('/03_TrainModels/', '/02_CreateRecords/')
    path_preprocess = fold.replace('Results', 'Folds')
    test_files = os.path.join(path_preprocess, 'test/*.tfrecord')

    new = multiband.Network()
    settings_path = os.path.join(path_run, 'all_settings.json')

    fold_name = fold.split('/')[-1]
    write_path = os.path.join('./Results', fold_name + '_Results.dat')

    # Get weights path
    run = path_run.split('/')[-1]
    weights_path = os.path.join(fold, 'Models', run)
    new.run_test(settings_path, test_files, weights_path, df_paths=write_path)

    result_path = os.path.join('./Results', fold.split('/')[-1] + '_Results.dat')

    # From the results file, read the data and compute the classification scores
    report_fold = plot.compute_classification_report(result_path)
    reports_folds.append(report_fold)
    # From the results file, read the data and compute the confusion matrix and the respective  labels
    cm_fold = plot.compute_confussion_matrices(result_path, labels=label_order)
    cm_folds.append(cm_fold)
    # Extract the regression metrics
    regression = plot.compute_regression(result_path, new.physical_params)
    regressions.append(regression)

print('Median')
median, delta_pos, delta_neg = plot.obtain_accumulated_regressions(regressions, metric='median')
print(median, delta_pos, delta_neg)

median, delta_up, delta_down = plot.obtain_accumulated_metrics(reports_folds, metric='median', label_order=label_order)
print(median)
print(delta_up)
print(delta_down)

print((median / 100).round(2))
print(np.round(median.loc['f1-score'].mean() / 100, 3))

statistic = 'median'
img_path = './' + survey + '_' + statistic + '_c+r.pdf'
plot.plot_confusion_matrix(cm_folds, labels_=label_order, survey=survey, statistic=statistic, save_path=img_path)

accuracies = [reports_folds[i]['accuracy'] for i in range(len(reports_folds))]
print(np.round(np.median(accuracies), 4), np.round(np.mean(accuracies), 4))
