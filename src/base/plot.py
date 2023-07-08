from sklearn.metrics import confusion_matrix, classification_report, r2_score, mean_squared_error, \
    mean_absolute_percentage_error
import numpy as np
import matplotlib.pyplot as plt
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


def compute_regression(path_fold, phys_params):
    """
    Get the masked predictions, and remove the -1 values.
    Report summary statistics, such as mean, std, median, 25 and 75 percentiles.
    """

    # Read the output file
    df = pd.read_csv(path_fold)

    # Extract the mask for each parameter
    param_results = []
    for param in phys_params:
        true_phys = 'True_' + param
        true_col = df[true_phys]

        pred_phys = 'Pred_' + param
        pred_col = df[pred_phys]

        # Extract the mask
        mask = true_col != -1
        masked_true = true_col[mask]
        masked_pred = pred_col[mask]

        # Compute masked R2
        masked_r2 = r2_score(masked_true, masked_pred)
        masked_rmse = mean_squared_error(masked_true, masked_pred, squared=False)
        masked_MAPE = mean_absolute_percentage_error(masked_true, masked_pred)
        # Compute the percentage error
        param_result = {'R2': masked_r2, 'RMSE': masked_rmse, 'MAPE': masked_MAPE}
        param_results.append(param_result)

    param_results = pd.DataFrame(param_results, index=phys_params)
    return param_results


def obtain_accumulated_regressions(regressions, metric='mean'):
    index_ = regressions[0].index
    columns_ = regressions[0].columns
    regressions_np = np.array([df.values for df in regressions])

    if metric == 'mean':
        mean = np.mean(regressions_np, axis=0)
        std = np.std(regressions_np, axis=0)

        mean = pd.DataFrame(mean, index=index_, columns=columns_)
        std = pd.DataFrame(std, index=index_, columns=columns_)
        return mean, std

    elif metric == 'median':
        summary = np.median(regressions_np, axis=0)
        q_25 = np.quantile(regressions_np, q=0.25, axis=0)
        q_75 = np.quantile(regressions_np, q=0.75, axis=0)

        summary_pos = q_75 - summary
        summary_neg = summary - q_25

        summary = pd.DataFrame(data=summary, index=index_, columns=columns_)
        summary_pos = pd.DataFrame(data=summary_pos, index=index_, columns=columns_)
        summary_neg = pd.DataFrame(data=summary_neg, index=index_, columns=columns_)
        return summary, summary_pos, summary_neg
    else:
        return None


def compute_confussion_matrices(path_fold, labels=None):
    # Read the output file
    df = pd.read_csv(path_fold)
    if labels is None:
        labels = [new.trans[i] for i in range(new.num_classes)]
    cm_fold = confusion_matrix(df.Class, df.Pred, labels=labels, normalize='true')

    return cm_fold


def compute_classification_report(path_fold):
    # Read the output file
    df = pd.read_csv(path_fold)
    # Compute the metrics per each fold
    report = classification_report(df.Class, df.Pred, output_dict=True, zero_division=0)
    return report


def obtain_accumulated_metrics(reports_, metric='mean', label_order=None):
    reports_np = 100 * np.array(
        [pd.DataFrame(i)[label_order].loc[['precision', 'recall', 'f1-score', 'support']] for i in reports_])
    reports_df = [100 * pd.DataFrame(i)[label_order].loc[['precision', 'recall', 'f1-score', 'support']] for i in
                  reports_]

    index_ = reports_df[0].index
    columns_ = reports_df[0].columns

    if metric == 'mean':
        summary = np.mean(reports_np, axis=0)
        summary_pos = np.std(reports_np, axis=0)
        summary_neg = None

        summary = pd.DataFrame(data=summary, index=index_, columns=columns_)
        summary_pos = pd.DataFrame(data=summary_pos, index=index_, columns=columns_)
        return summary, summary_pos
    elif metric == 'median':
        summary = np.median(reports_np, axis=0)
        q_25 = np.quantile(reports_np, q=0.25, axis=0)
        q_75 = np.quantile(reports_np, q=0.75, axis=0)

        summary_pos = q_75 - summary
        summary_neg = summary - q_25

        summary = pd.DataFrame(data=summary, index=index_, columns=columns_)
        summary_pos = pd.DataFrame(data=summary_pos, index=index_, columns=columns_)
        summary_neg = pd.DataFrame(data=summary_neg, index=index_, columns=columns_)
        return summary, summary_pos, summary_neg
    else:
        return None


def plot_confusion_matrix(cm_folds,
                          labels_,
                          title='Confusion matrix',
                          statistic='mean',
                          survey=None,
                          cmap=plt.cm.Greens,
                          save_path=None,
                          nep=0,
                          ):
    """ This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm_folds = 100 * np.array(cm_folds)
    font = {'family': 'serif',
            'weight': 'normal',
            'serif': ['Times New Roman'] + plt.rcParams['font.serif']}
    plt.clf()
    np.set_printoptions(precision=2)

    fig = plt.figure(figsize=(10, 10), dpi=250)
    ax = plt.gca()

    small_size = 15
    medium_size = 19
    bigger_size = 25

    num_classes = len(labels_)

    plt.rc('font', size=small_size, **font)  # controls default text sizes
    plt.rc('axes', titlesize=24)  # fontsize of the axes title
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=small_size)  # legend fontsize
    plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

    # plt.title(title)
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, [i for i in labels_], rotation=45)
    plt.yticks(tick_marks, [i for i in labels_])

    if statistic == 'mean':
        center_cm = np.mean(cm_folds, axis=0)
        std_cm = np.std(cm_folds, axis=0)

        cm_test = 100.0 * np.array(cm_folds)

        cm_pos = std_cm
        cm_neg = std_cm

    elif statistic == 'median':
        center_cm = np.median(cm_folds, axis=0)
        q_25 = np.quantile(cm_folds, 0.25, axis=0)
        q_75 = np.quantile(cm_folds, 0.75, axis=0)

        cm_pos = q_75 - center_cm
        cm_neg = center_cm - q_25

    else:
        print('Error: Not implemented')
        return None

    im = ax.imshow(center_cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
    fig.canvas.draw()
    tt = fig.gca().transData.inverted()

    for i, j in itertools.product(range(num_classes), range(num_classes)):
        if center_cm[i, j] >= 1e-1:
            center = round(center_cm[i, j], 1)
            pos = round(cm_pos[i, j], 1)
            neg = round(cm_neg[i, j], 1)

            super_text = f'${pos}}}$'
            middle = f'${center}$'
            sub_text = f'${neg}}}$'

            # Small shift in x to accommodate the super/sub scripts
            mid_text = plt.text(j - 0.075, i + 0.075, middle, horizontalalignment="center",
                                fontdict={'fontsize': small_size}, bbox={'alpha': 0.0, 'lw': 0, 'pad': 0}
                                , color="white" if center_cm[i, j] > 80 else "black")

            new_coords = tt.transform_bbox(mid_text.get_window_extent()).get_points()

            if statistic == 'mean':
                sub_text = f''

            sup_text = plt.text(new_coords[1, 0] - 0.115, new_coords[1, 1] - 0.01, super_text,
                                verticalalignment='bottom', horizontalalignment="left",
                                fontdict={'fontsize': small_size - 3}
                                , color="white" if center_cm[i, j] > 80 else "black")
            bot_text = plt.text(new_coords[1, 0] - 0.115, new_coords[0, 1] - 0.01, sub_text, verticalalignment='top',
                                horizontalalignment="left", fontdict={'fontsize': small_size - 3}
                                , color="white" if center_cm[i, j] > 80 else "black")

        elif center_cm[i, j] > 0:
            plt.text(j, i, '$<0.01$', horizontalalignment="center"
                     , color="black", fontdict={'size': small_size - 2, 'weight': 'normal'})

    if survey == 'Gaia':
        # Horizontal Upper RR Lyrae
        plt.hlines(2.5, 2.5, 4.5, alpha=1, lw=2, color='black')
        # Horizontal Lower RR Lyrae
        plt.hlines(4.5, 2.5, 4.5, alpha=1, lw=2, color='black')
        # Vertical Left RR Lyrae
        plt.vlines(2.5, 2.5, 4.5, alpha=1, lw=2, color='black')
        # Vertical Right RR Lyrae
        plt.vlines(4.5, 2.5, 4.5, alpha=1, lw=2, color='black')

        # Horizontal  CEP
        # Horizontal Upper CEP
        plt.hlines(1.5, -0.5, 1.5, alpha=1, lw=2, color='black')
        # Vertical Left CEP
        # Vertical Right CEP
        plt.vlines(1.5, -0.5, 1.5, alpha=1, lw=2, color='black')
    elif survey == 'ZTF':
        # Horizontal Lower Transients
        plt.hlines(0.5, -0.5, 0.5, alpha=1, lw=2, color='black')
        # Vertical Right Transients
        plt.vlines(0.5, 0.5, -0.5, alpha=1, lw=2, color='black')

        # Horizontal Lower Stochastic
        plt.hlines(5.5, 0.5, 5.5, alpha=1, lw=2, color='black')
        # Horizontal Upper Stochastic
        plt.hlines(0.5, 0.5, 5.5, alpha=1, lw=2, color='black')
        # Vertical Left Stochastic
        plt.vlines(0.5, 0.5, 5.5, alpha=1, lw=2, color='black')
        # Vertical Right Stochastic
        plt.vlines(5.5, 0.5, 5.5, alpha=1, lw=2, color='black')

        # Horizontal Upper Periodic
        plt.hlines(5.5, 5.5, 8.5, alpha=1, lw=2, color='black')
        # Vertical Left Periodic
        plt.vlines(5.5, 5.5, 8.5, alpha=1, lw=2, color='black')
    elif survey == 'PanStarrs':
        # Horizontal Lower RRL
        plt.hlines(1.5, 1.5, 4.5, alpha=1, lw=2, color='black')
        # Horizontal Upper RRL
        plt.hlines(4.5, 1.5, 4.5, alpha=1, lw=2, color='black')
        # Vertical Left RRL
        plt.vlines(1.5, 1.5, 4.5, alpha=1, lw=2, color='black')
        # Vertical Right RRL
        plt.vlines(4.5, 1.5, 4.5, alpha=1, lw=2, color='black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
