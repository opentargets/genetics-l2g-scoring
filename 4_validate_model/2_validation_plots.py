#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Analyses predictions
'''

import sys
import os
import pandas as pd
from pprint import pprint
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sklearn.metrics as skmet
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import argparse

def main():

    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', 500)

    # Parse args
    global args
    args = parse_args()

    # Allow plots or classification report to be switched off
    do_plots = True
    do_clf_report = True

    # Map feature names
    feature_name_map = {
        'dist_foot_min': 'Distance to gene [min]',
        'dist_foot_min_nbh': 'Distance to gene [min; nbh]',
        'dist_foot_ave': 'Distance to gene [average]',
        'dist_foot_ave_nbh': 'Distance to gene [average; nbh]',
        'dist_tss_min': 'Distance to TSS [min]',
        'dist_tss_min_nbh': 'Distance to TSS [min; nbh]',
        'dist_tss_ave': 'Distance to TSS [average]',
        'dist_tss_ave_nbh': 'Distance to TSS [average; nbh]',
        'eqtl_coloc_llr_max': 'eQTL coloc [max]',
        'eqtl_coloc_llr_max_neglogp': 'eQTL coloc [pval]',
        'eqtl_coloc_llr_max_nbh': 'eQTL coloc [max; nbh]',
        'pqtl_coloc_llr_max': 'pQTL coloc [max]',
        'pqtl_coloc_llr_max_neglogp': 'pQTL coloc [pval]',
        'pqtl_coloc_llr_max_nbh': 'pQTL coloc [max; nbh]',
        'eqtl_pics_clpp_max': 'eQTL PICS [max]',
        'eqtl_pics_clpp_max_neglogp': 'eQTL PICS [pval]',
        'eqtl_pics_clpp_max_nhb': 'eQTL PICS [max; nbh]',
        'pqtl_pics_clpp_max': 'pQTL PICS [max]',
        'pqtl_pics_clpp_max_neglogp': 'pQTL PICS [pval]',
        'pqtl_pics_clpp_max_nhb': 'pQTL PICS [max; nbh]',
        'dhs_prmtr_max': 'DHS-promoter correlation [max]',
        'dhs_prmtr_max_nbh': 'DHS-promoter correlation [max; nbh]',
        'dhs_prmtr_ave': 'DHS-promoter correlation [average]',
        'dhs_prmtr_ave_nbh': 'DHS-promoter correlation [average; nbh]',
        'enhc_tss_max': 'Enhancer-TSS correlation [max]',
        'enhc_tss_max_nbh': 'Enhancer-TSS correlation [max; nbh]',
        'enhc_tss_ave': 'Enhancer-TSS correlation [average]',
        'enhc_tss_ave_nbh': 'Enhancer-TSS correlation [average; nbh]',
        'pchic_max': 'PCHiC score [max]',
        'pchic_max_nbh': 'PCHiC score [max; nbh]',
        'pchic_ave': 'PCHiC score [average]',
        'pchic_ave_nbh': 'PCHiC score [average; nbh]',
        'pchicJung_max': 'PCHiC Jung score [max]',
        'pchicJung_max_nbh': 'PCHiC Jung score [max; nbh]',
        'pchicJung_ave': 'PCHiC Jung score [average]',
        'pchicJung_ave_nbh': 'PCHiC Jung score [average; nbh]',
        'vep_credset_max': 'VEP score [max]',
        'vep_credset_max_nbh': 'VEP score [max; nbh]',
        'vep_ave': 'VEP score [average]',
        'vep_ave_nbh': 'VEP score [average; nbh]',
        'polyphen_credset_max': 'PolyPhen score [max]',
        'polyphen_credset_max_nbh': 'PolyPhen score [max; nbh]',
        'polyphen_ave': 'PolyPhen score [average]',
        'polyphen_ave_nbh': 'PolyPhen score [average; nbh]',
        'count_credset_95': 'Credible set size',
        'has_sumstats': 'Whether summary stats available',
        'gene_count_lte_50k': 'Gene count within 50kb',
        'gene_count_lte_100k': 'Gene count within 100kb',
        'gene_count_lte_250k': 'Gene count within 250kb',
        'gene_count_lte_500k': 'Gene count within 500kb',
        'proteinAttenuation': 'Protein attenuation'
    }

    # Define metrics for the classification report
    metrics = {
        # Scores requiring predictions only
        'y_pred': {
            'accuracy_score': skmet.accuracy_score,
            'balanced_accuracy_score': skmet.balanced_accuracy_score,
            # 'cohen_kappa_score': skmet.cohen_kappa_score,
            'f1_score': skmet.f1_score,
            # 'hamming_loss': skmet.hamming_loss,
            # 'jaccard_score': skmet.jaccard_score,
            # 'log_loss': skmet.log_loss,
            # 'matthews_corrcoef': skmet.matthews_corrcoef,
            'precision_score': skmet.precision_score,
            'recall_score': skmet.recall_score,
            # 'zero_one_loss': skmet.zero_one_loss
        },
        # Scores requiring class probabilities
        'y_proba': {
            'average_precision_score': skmet.average_precision_score,
            # 'brier_score_loss': skmet.brier_score_loss,
            'roc_auc_score': skmet.roc_auc_score
        }
    }

    #
    # Load --------------------------------------------------------------------
    #

    # Load predictions
    pred = pd.read_parquet(args.in_pred)

    # Load feature importance information
    with open(args.in_ftimp, 'r') as in_h:
        ft_imp = json.load(in_h)

    #
    # Check predictions -------------------------------------------------------
    #

    # Count how many loci in total
    top_loci_count = (
        pred
        .loc[:, ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']]
        .drop_duplicates()
        .shape[0]
    )

    # Count how many predictions per classifier
    pred_counts = (
        pred
        .groupby(['clf_classifier_name', 'clf_feature_name', 'clf_gold_standard_set'])
        .study_id
        .size()
        .reset_index()
    )

    # Make sure the correct number of loci have been predicted
    # print('Warning: skipping assertation!\n')
    assert (pred_counts['study_id'] == top_loci_count).all()

    #
    # Process predictions -------------------------------------------
    #

    # Initiate classification report
    clf_report = []

    # Group predictions
    pred_grp = pred.groupby([
        'clf_classifier_name',
        'clf_feature_name',
        'clf_gold_standard_set'])
    
    # Iterate over training groups
    for (clf, ft, gs_training), group in pred_grp:

        print('\nProcessing', clf, ft, gs_training, '...')

        # Make testing gold-standard sets. This is the same as in 1_train_models.py
        gs_sets = {
            #'high_medium_low': group['gs_confidence'].isin(['High', 'Medium', 'Low']),
            'high_medium': group['gs_confidence'].isin(['High', 'Medium']),
            #'high': group['gs_confidence'].isin(['High']),
            #'sumstat_only': group['has_sumstats'] == 1,
            #'progem': group['gs_set'] == 'ProGeM',
            #'t2d': group['gs_set'] == 'T2D Knowledge Portal ',
            #'chembl_all': group['gs_set'].isin(['ChEMBL_IV', 'ChEMBL_III', 'ChEMBL_II']),
            #'chembl_excl_II': group['gs_set'].isin(['ChEMBL_IV', 'ChEMBL_III']),
            #'fauman_twitter': group['gs_set'].isin(['Eric Fauman Twitter']),
            #'ot_curated': group['gs_set'].isin(['otg_curated_191108']),
        }

        # Iterate over testing gold-standard sets
        for gs_test, gs_set in gs_sets.items():

            # Subset rows of the group dataset
            group_subset = group.loc[gs_set, :]

            # Skip if empty
            if group_subset.shape[0] == 0:
                print('Warning: gs_test={} subset is empty, skipping...'.format(gs_test))
                continue

            #
            # Make plots ----------------------------------------------------------
            #

            if do_plots:

                # Make outname
                out_name = '{}-{}-{}-{}.figure.png'.format(clf, ft, gs_training, gs_test)
                out_path = os.path.join(*[
                    args.out_plotdir,
                    clf,
                    'training=' + gs_training,
                    out_name
                ])
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Initiate figure
                fig = plt.figure(figsize=(18, 18), dpi=300)
                fig.suptitle(' '.join([
                    clf,
                    'training=' + gs_training,
                    'testing=' + gs_test,
                    ft]))
                grid_spec = gridspec.GridSpec(7, 8)
                grid_spec.update(wspace=1)#, hspace=0.5)

                # Make plot data
                y_true = group_subset['gold_standard_status'].tolist()
                y_pred = group_subset['y_pred'].tolist()
                y_proba = group_subset['y_proba'].tolist()
                fold_data = []
                for fold_name, fold_grp in group_subset.groupby('clf_fold_name'):
                    fold_data.append({
                        'y_true': fold_grp['gold_standard_status'].tolist(),
                        'y_pred': fold_grp['y_pred'].tolist(),
                        'y_proba': fold_grp['y_proba'].tolist(),
                        'fold_name': fold_name,
                        'ft_imp': ft_imp[clf][ft][gs_training][fold_name]['feature_importances']
                    })

                # Plot precision-recall curve
                ax_prc = plt.subplot(grid_spec[0:2, 0:4])
                ax_prc = plot_precision_recall_curve(
                    y_true=y_true,
                    probas_pred=y_proba,
                    ax=ax_prc,
                    subtitle=None,
                    fold_data=fold_data)
                fig.add_subplot(ax_prc)

                # Plot ROC curve
                ax_roc = plt.subplot(grid_spec[0:2, 4:8])
                ax_roc = plot_roc_curve(
                    y_true=y_true,
                    probas_pred=y_proba,
                    ax=ax_roc,
                    subtitle=None,
                    fold_data=fold_data)
                fig.add_subplot(ax_roc)

                # Plot calibration curve
                ax_cal_curve = plt.subplot(grid_spec[3:5, 0:4])
                ax_cal_hist = plt.subplot(grid_spec[5, 0:4])
                ax_cal_curve, ax_cal_hist = plot_calibration_curve(
                    y_true=y_true,
                    probas_pred=y_proba,
                    ax_curve=ax_cal_curve,
                    ax_hist=ax_cal_hist,
                    subtitle=None,
                    fold_data=fold_data)
                fig.add_subplot(ax_cal_curve)
                fig.add_subplot(ax_cal_hist)

                # Feature importances
                ax_ftimp = plt.subplot(grid_spec[3:7, 5:8])
                ax_ftimp = plot_feature_importances(
                    fold_data=fold_data,
                    feature_names=ft_imp[clf][ft]['feature_names'],
                    ax=ax_ftimp,
                    subtitle=None)
                fig.add_subplot(ax_ftimp)
                
                # Plot and save figure
                plt.savefig(out_path)
                plt.close()

            #
            # Make classification report ------------------------------------------
            #

            if do_clf_report:

                # Initiate output for this row
                clf_row = {
                    'clf_name': clf,
                    'feature_set': ft,
                    'goldstandard_training': gs_training,
                    'goldstandard_testing': gs_test
                }

                # Calculate metrics for y_pred and y_proba
                for metric_type in metrics:
                    for metric in metrics[metric_type]:
                        # Calc
                        score = metrics[metric_type][metric](
                            group_subset['gold_standard_status'].tolist(),
                            group_subset[metric_type].tolist())
                        # Add to report
                        clf_row[metric] = score
                
                # Calculate confusion matrix
                tn, fp, fn, tp = skmet.confusion_matrix(
                    group_subset['gold_standard_status'].tolist(),
                    group_subset['y_pred'].tolist()
                ).ravel()
                clf_row['true_negatives'] = tn
                clf_row['false_positives'] = fp
                clf_row['false_negatives'] = fn
                clf_row['true_positives'] = tp

                # Add derivations from confusion matrix
                clf_row['tpr/sensitivity'] = tp / (tp + fn)
                clf_row['tnr/specificity'] = tn / (tn + fp)
                clf_row['fpr/fallout'] = fp / (fp + tn)
                clf_row['fnr/missrate'] = fn / (fn + tp)
                clf_row['fdr'] = fp / (fp + tp)

                # Support (number of lables for each class)
                clf_row['support_1'] = (
                    group_subset['gold_standard_status'] == 1
                ).sum()
                clf_row['support_0'] = (
                    group_subset['gold_standard_status'] == 0
                ).sum()
            
                clf_report.append(clf_row)

                # break
    
    #
    # Write classification report ---------------------------------------------
    #

    print('Writing classification report...')

    # Convert to df
    clf_df = pd.DataFrame(clf_report)
    
    # Write
    os.makedirs(os.path.dirname(args.out_report), exist_ok=True)
    clf_df.to_csv(args.out_report, sep='\t', index=True, index_label='idx')
    
    return 0

def plot_precision_recall_curve(y_true, probas_pred, ax, subtitle,
    fold_data=None):
    ''' Makes a precision-recall curve
    Params:
        y_true (list): true classes
        probas_pred (list): prediction probability for classes
        subtitle (str): Plot subtitle
        ax (matplotlib.ax): axis
        fold_data (list): data for each individual folds
    Returns:
        ax
    '''

    # Plot main result
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)
    average_precision = average_precision_score(y_true, probas_pred)
    ax.step(recall, precision, color='b', alpha=0.8,
            where='post', label='Overall AP = {:.2f}'.format(average_precision))

    # Plot each fold
    if fold_data:
        for fold in fold_data:
            precision, recall, _ = precision_recall_curve(
                fold['y_true'], fold['y_proba'])
            average_precision = average_precision_score(
                fold['y_true'], fold['y_proba'])
            ax.step(recall, precision, alpha=0.2,
                where='post', label='{0} AP = {1:.2f}'.format(
                    fold['fold_name'], average_precision)
                )

    # Add labels
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="best", prop={'size': 6})
    if subtitle:
        ax.set_title('Precision-Recall Curve\n{}'.format(title))
    else:
        ax.set_title('Precision-Recall Curve')

    return ax

def plot_roc_curve(y_true, probas_pred, ax, subtitle=None,
        fold_data=None):
    ''' Makes ROC curve
    Params:
        y_true (list): true classes
        probas_pred (list): prediction probability for classes
        subtitle (str): Plot subtitle
        ax (matplotlib.figure): ax
        fold_data (list): data for each individual folds
    Returns:
        ax
    '''

    ax.plot([0, 1], [0, 1], "k:")

    # Plot main result
    fpr, tpr, _ = roc_curve(y_true, probas_pred)
    score = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='b', alpha=0.8,
            label='Overall AUC = {:.2f}'.format(score))

    # Plot each fold
    for i, fold in enumerate(fold_data):
        fpr, tpr, _ = roc_curve(
            fold['y_true'], fold['y_proba'])
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.2,
            label='Fold {0} AUC = {1:.2f}'.format(
                i + 1, score)
            )

    # Add labels
    ax.set_xlabel('False Positive Rate', fontsize=16)
    ax.set_ylabel('True Positive Rate', fontsize=16)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc="best", prop={'size': 12})
    if subtitle:
        ax.set_title('Receiver Operating Characteristic\n{}'.format(title), fontdict={'fontsize': 18})
    else:
        ax.set_title('Receiver Operating Characteristic', fontdict={'fontsize': 18})

    return ax

def plot_calibration_curve(y_true, probas_pred, ax_curve, ax_hist,
        subtitle, fold_data=None, n_bins=10):
    ''' Makes a calibration curve
    Params:
        y_true (list): true classes
        probas_pred (list): prediction probability for classes
        subtitle (str): Plot subtitle
        ax_curve (matplotlib.ax): axis to plot the curve on
        ax_hist (matplotlib.ax): axis to plot the histogram on
        fold_data (list): data for each individual folds
    Returns:
        ax
    '''

    # Plot perfectly calibrated line
    ax_curve.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot main result
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, probas_pred, n_bins=n_bins)
    ax_curve.plot(mean_predicted_value, fraction_of_positives, "s-",
        label='Overall', alpha=0.8, color='b')
    ax_hist.hist(probas_pred, range=(0, 1), bins=10, label='Overall',
                histtype="step", lw=2, color='b', alpha=0.8)

    # Plot each fold
    if fold_data:
        for i, fold in enumerate(fold_data):
            fraction_of_positives, mean_predicted_value = \
            calibration_curve(fold['y_true'], fold['y_proba'],
                n_bins=n_bins)
            ax_curve.plot(mean_predicted_value, fraction_of_positives, "s-",
                label=f'Fold {i+1}', alpha=0.2)
            ax_hist.hist(probas_pred, range=(0, 1), bins=10,
                label=f'Fold {i+1}',
                histtype="step", lw=2, alpha=0.2)
            
    # Add labels to histogram
    ax_hist.set_xlabel('Mean predicted value', fontsize=16)
    ax_hist.set_ylabel('Count', fontsize=16)
    ax_hist.set_xlim([0.0, 1.0])
    # Add labels to curve
    ax_curve.set_xticklabels([])
    ax_curve.set_ylabel('Fraction of positives', fontsize=16)
    ax_curve.set_ylim([0.0, 1.05])
    ax_curve.set_xlim([0.0, 1.0])
    ax_curve.legend(loc="best", prop={'size': 12})
    if subtitle:
        ax_curve.set_title('Calibration Curve\n{}'.format(title), fontdict={'fontsize': 18})
    else:
        ax_curve.set_title('Calibration Curve', fontdict={'fontsize': 18})

    return ax_curve, ax_hist

def plot_feature_importances(fold_data, feature_names, ax,  subtitle=None):
    ''' Makes a plot of feature importances
    Params:
        fold_data (list of dicts): data for each individual folds
        feature_names (list): list of feature names
        ax (matplotlib.figure): ax
        subtitle (str): Plot subtitle
    Returns:
        ax
    '''

    bar_width = 0.5

    # Calculate mean feature importance across folds
    ft_imps = np.array([fold['ft_imp'] for fold in fold_data])
    ft_imps_mean = list(np.mean(ft_imps, axis=0))

    # Sort from high to low
    srt_idx = np.argsort(ft_imps_mean)#[::-1]
    ft_imps_mean = np.array(ft_imps_mean)[srt_idx]
    feature_names = np.array(feature_names)[srt_idx]
    
    # Plot main result
    x_min_pos = [x - bar_width for x in range(len(ft_imps_mean))]
    x_max_pos = [x + bar_width for x in range(len(ft_imps_mean))]
    ax.vlines(ft_imps_mean, x_min_pos, x_max_pos, label='Overall', colors='b', alpha=0.8)

    # Plot each fold
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    #           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, fold in enumerate(fold_data):
        fold_values = np.array(fold['ft_imp'])[srt_idx]
        ax.vlines(
            fold_values,
            x_min_pos,
            x_max_pos,
            # label=f'Fold {i+1}',
            label=f'Folds',
            color='grey',
            alpha=0.2
        )

    # Add horizontal lines
    ax.axvline(x=np.max(ft_imps_mean), linestyle='--', alpha=0.2)
    ax.axvline(x=np.min(ft_imps_mean), linestyle='--', alpha=0.2)

    # Add vertical lines
    for pos in x_min_pos[0:1] + x_max_pos:
        ax.axhline(pos, linestyle='-', alpha=0.1)

    # Add labels
    ax.set_xlabel('Importance', fontsize=16)
    ax.set_yticks(range(len(ft_imps_mean)))
    ax.set_yticklabels(feature_names, fontsize=12)
    # ax.xaxis.set_tick_params(rotation=90)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 12})
    if subtitle:
        ax.set_title('Feature Importances\n{}'.format(title), fontdict={'fontsize': 18})
    else:
        ax.set_title('Feature Importances', fontdict={'fontsize': 18})

    return ax

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_pred', metavar="<parquet>", help="Input parquet containing predictions", type=str, required=True)
    parser.add_argument('--in_ftimp', metavar="<str>", help="Input json containing feature importances", type=str, required=True)
    # Outputs
    parser.add_argument('--out_plotdir', metavar="<dir>", help="Output directory to write plots", type=str, required=True)
    parser.add_argument('--out_report', metavar="<file>", help="Out path for classification report", type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
