#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Performs xgboost classification:
- Outer CV using grouped chromosomes
- Inner CV used for Hyper-parameter tuning
- Save best trained model for predictions
'''

import sys
import os
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import make_scorer, balanced_accuracy_score
from pprint import pprint
import joblib
import xgboost as xgb
import scipy.stats as sps
import warnings
import argparse

def main():

    pd.options.mode.chained_assignment = None
    pd.set_option('display.max_columns', 500)

    # Parse args
    global args
    args = parse_args()
    inner_cv_metric = balanced_accuracy_score
    all_chroms = [str(chrom) for chrom in range(1, 23)] + ['X']

    #
    # Prepare data ------------------------------------------------------------
    #

    # Load
    data = pd.read_parquet(in_data)

    # Recode True/False
    data = data.replace({True: 1, False: 0})

    # Make scorer
    inner_cv_scorer = make_scorer(inner_cv_metric)

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=random_state)

    # Create output folder
    os.makedirs(out_dir, exist_ok=True)

    #
    # Define different features, classifiers, parameters, gold-standards ------
    #

    # Make feature sets
    feature_sets = make_feature_sets()

    # Make dict of classifiers
    classifiers = {
        ('xgboost', 'xgboost'):
            xgb.XGBClassifier(
                booster="gbtree", # gbtree|dart
                objective="binary:logistic",
                random_state=random_state,
                n_jobs=1
            )
    }

    # Make parameter grid to match classifier types above
    param_grid = {
        'xgboost': {
            'n_estimators': sps.randint(100, 1000), # Same as nrounds
            'eta': sps.uniform(0, 0.4),
            'min_child_weight': sps.uniform(0, 10),
            'gamma': sps.expon(0, 0.1),
            'subsample': sps.uniform(0.4, 0.6), # [0.4, 1]
            'colsample_bytree': sps.uniform(0.4, 0.6), # [0.4, 1]
            'colsample_bylevel': sps.uniform(0.4, 0.6), # [0.4, 1]
            'max_depth': sps.randint(10, 50), # 100 is too high
            'learning_rate': sps.expon(0, 0.01),
            'reg_lambda': sps.uniform(0, 100),
            'max_delta_step': [1] # Usually this parameter is not needed (0), but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
        }
    }

    # Make gold-standard sets. Only training on medium and high for production
    # model
    gs_sets = {
        'high_medium': data['gs_confidence'].isin(['High', 'Medium']),
        # 'high_medium_low': data['gs_confidence'].isin(['High', 'Medium', 'Low']),
        # 'high': data['gs_confidence'].isin(['High']),
        # 'sumstat_only': data['has_sumstats'] == 1,
        # 'progem': data['gs_set'] == 'ProGeM',
        # 't2d': data['gs_set'] == 'T2D Knowledge Portal ',
        # 'chembl_all': data['gs_set'].isin(['ChEMBL_IV', 'ChEMBL_III', 'ChEMBL_II']),
        # 'chembl_excl_II': data['gs_set'].isin(['ChEMBL_IV', 'ChEMBL_III']),
        # 'fauman_twitter': data['gs_set'].isin(['Eric Fauman Twitter']),
        # 'ot_curated': data['gs_set'].isin(['otg_curated_191108']),
    }

    #
    # Train models ------------------------------------------------------------
    #

    # Iterate over classifiers
    for (clf_type, clf_name), clf in classifiers.items():
        # Iterate over feature sets
        for ft_name, ft in feature_sets.items():
            # Iterate over gold-standard sets
            for gs_name, gs_set in gs_sets.items():
                
                # Subset rows of the data
                data_subset = data.loc[gs_set, :]

                # Get chromosome groups for outer CV
                outer_cv_chrom_groups = get_cv_groups(
                    df=data_subset,
                    num_folds=num_outer_folds,
                    group_col='chrom',
                    outcome_col=outcome_col,
                    all_groups=all_chroms
                )

                # Perfrom leave-one-group-out (logo) outer CV
                for outer_fold_num, test_group in enumerate(outer_cv_chrom_groups):

                    # Make dict of run info
                    run_info = {
                        'classifier_name': clf_name,
                        'feature_name': ft_name,
                        'features': ft,
                        'gold_standard_set': gs_name,
                        'fold_num': outer_fold_num,
                        'fold_test_chroms': test_group,
                        'fold_name': 'fold{0}={1}'.format(
                            outer_fold_num,
                            '|'.join(test_group)
                        )
                    }
                    pprint(run_info)

                    # Create output name and skip if existing
                    run_name = '-'.join([
                        clf_name, ft_name, gs_name, str(outer_fold_num)]).replace(' ', '_')
                    out_path = os.path.join(
                        out_dir,
                        '{}.model.joblib.gz'.format(run_name)
                    )
                    if os.path.exists(out_path):
                        print(f'\nSkipping, model output exists: {out_path}\n')
                        continue

                    # Get training and test indexes
                    train_index = ~(data_subset['chrom'].isin(test_group))
                    test_index = data_subset['chrom'].isin(test_group)

                    # Split data into training and test
                    X_train = data_subset.loc[train_index, ft]
                    y_train = data_subset.loc[train_index, outcome_col]
                    X_test = data_subset.loc[test_index, ft]
                    y_test = data_subset.loc[test_index, outcome_col]

                    # Make inner CV for model selection
                    inner_grid_search = RandomizedSearchCV(
                        estimator=clf,
                        param_distributions=param_grid[clf_type],
                        scoring=inner_cv_scorer,
                        n_jobs=n_jobs,
                        cv=num_inner_folds,
                        verbose=1,
                        n_iter=cv_n_iter
                    )
                    
                    # Fit inner CV
                    inner_grid_search.fit(
                        X_train,
                        y_train,
                        early_stopping_rounds=early_stopping_rounds,
                        eval_metric=eval_metric,
                        eval_set=[(X_test, y_test)],
                        verbose=0
                    )

                    # Get outer score
                    y_pred = inner_grid_search.best_estimator_.predict(X_test)
                    y_proba = inner_grid_search.best_estimator_.predict_proba(X_test)[:, 1]
                    outer_score = inner_cv_metric(y_test, y_pred)

                    # Print inner and outer scores
                    print('Best inner score:', inner_grid_search.best_score_)
                    print('Outer score:', outer_score)
                    print()

                    # Create object to save
                    output = {
                        'run_info': run_info,
                        'model': inner_grid_search,
                        'scores': {
                            'best inner ap': inner_grid_search.best_score_,
                            'outer_ap': outer_score
                        }
                    }

                    # Persist model
                    joblib.dump(output, out_path, compress=True)

                # sys.exit()

    return 0

def get_cv_groups(df, num_folds, group_col, outcome_col, all_groups):
    ''' Groups gold-standards into equally sized chunks for
        CV, based on group_col
    Returns:
        a list of sets containing group labels
    '''

    # Count number of gold-standard positives per group
    group_counts = (
        df
        .query('{} == 1'.format(outcome_col))
        .groupby(group_col)
        .size()
        .to_dict()
    )

    # Add zero sized groups to group counts
    for group_name in all_groups:
        if group_name not in group_counts:
            group_counts[group_name] = 0

    # Setup counter and list of lists to hold group names
    fold_counts = [0 for x in range(num_folds)]
    fold_groups = [set([]) for x in range(num_folds)]

    # Iteratively add group to the smallest list
    for group_name, group_count in group_counts.items():
        # Get index of smallest fold
        i = fold_counts.index(min(fold_counts))
        # Add group to that fold
        fold_groups[i].add(group_name)
        fold_counts[i] = fold_counts[i] + group_count

    return fold_groups

def make_feature_sets():
    ''' Make different sets of features to run the model on
    
    List of all features (October 2019):

    'dhs_prmtr_max',
    'dhs_prmtr_max_nbh', 'dhs_prmtr_ave', 'dhs_prmtr_ave_nbh',
    'enhc_tss_max', 'enhc_tss_max_nbh', 'enhc_tss_ave', 'enhc_tss_ave_nbh',
    'eqtl_coloc_llr_max', 'eqtl_coloc_llr_max_neglogp',
    'eqtl_coloc_llr_max_nbh', 'pqtl_coloc_llr_max',
    'pqtl_coloc_llr_max_neglogp', 'pqtl_coloc_llr_max_nbh', 'pchic_max',
    'pchic_max_nbh', 'pchic_ave', 'pchic_ave_nbh', 'pchicJung_max',
    'pchicJung_max_nbh', 'pchicJung_ave', 'pchicJung_ave_nbh',
    'eqtl_pics_clpp_max', 'eqtl_pics_clpp_max_neglogp',
    'eqtl_pics_clpp_max_nhb', 'pqtl_pics_clpp_max',
    'pqtl_pics_clpp_max_neglogp', 'pqtl_pics_clpp_max_nhb',
    'vep_credset_max', 'vep_credset_max_nbh', 'vep_ave', 'vep_ave_nbh',
    'polyphen_credset_max', 'polyphen_credset_max_nbh', 'polyphen_ave',
    'polyphen_ave_nbh', 'interlocus_string_bystudy_unweighted',
    'interlocus_string_bystudy_targetweighted',
    'interlocus_string_bystudy_unweighted_locusweight',
    'interlocus_string_bystudy_targetweighted_locusweight',
    'interlocus_string_byefo_unweighted',
    'interlocus_string_byefo_targetweighted',
    'interlocus_string_byefo_unweighted_locusweight',
    'interlocus_string_byefo_targetweighted_locusweight',
    'count_credset_95', 'has_sumstats', 'dist_foot_sentinel',
    'dist_foot_sentinel_nbh', 'dist_foot_min', 'dist_foot_min_nbh',
    'dist_foot_ave', 'dist_foot_ave_nbh', 'dist_tss_sentinel',
    'dist_tss_sentinel_nbh', 'dist_tss_min', 'dist_tss_min_nbh',
    'dist_tss_ave', 'dist_tss_ave_nbh', 'gene_count_lte_50k',
    'gene_count_lte_100k', 'gene_count_lte_250k', 'gene_count_lte_500k',
    'proteinAttenuation'

    '''

    #
    # Specify feature groups --------------------------------------------------
    #

    distance = [
        # 'dist_foot_sentinel', 'dist_foot_sentinel_nbh', 
        # 'dist_tss_sentinel', 'dist_tss_sentinel_nbh',
        'dist_foot_min', 'dist_foot_min_nbh',
        'dist_foot_ave', 'dist_foot_ave_nbh',
        'dist_tss_min', 'dist_tss_min_nbh',
        'dist_tss_ave', 'dist_tss_ave_nbh'
    ]
    molecularQTL = [
        'eqtl_coloc_llr_max', 'eqtl_coloc_llr_max_neglogp',
        'eqtl_coloc_llr_max_nbh', 'pqtl_coloc_llr_max',
        'pqtl_coloc_llr_max_neglogp', 'pqtl_coloc_llr_max_nbh',
        'eqtl_pics_clpp_max', 'eqtl_pics_clpp_max_neglogp',
        'eqtl_pics_clpp_max_nhb', 'pqtl_pics_clpp_max',
        'pqtl_pics_clpp_max_neglogp', 'pqtl_pics_clpp_max_nhb',
    ]
    interaction = [
        'dhs_prmtr_max', 'dhs_prmtr_max_nbh', 'dhs_prmtr_ave', 'dhs_prmtr_ave_nbh',
        'enhc_tss_max', 'enhc_tss_max_nbh', 'enhc_tss_ave', 'enhc_tss_ave_nbh',
        'pchic_max', 'pchic_max_nbh', 'pchic_ave', 'pchic_ave_nbh', 'pchicJung_max',
        'pchicJung_max_nbh', 'pchicJung_ave', 'pchicJung_ave_nbh',
    ]
    pathogenicity = [
        'vep_credset_max', 'vep_credset_max_nbh', 'vep_ave', 'vep_ave_nbh',
        'polyphen_credset_max', 'polyphen_credset_max_nbh', 'polyphen_ave',
        'polyphen_ave_nbh',
    ]
    # interlocus = [
    #     # 'interlocus_string_bystudy_unweighted',
    #     # 'interlocus_string_bystudy_targetweighted',
    #     # 'interlocus_string_bystudy_unweighted_locusweight',
    #     'interlocus_string_bystudy_targetweighted_locusweight',
    #     # 'interlocus_string_byefo_unweighted',
    #     # 'interlocus_string_byefo_targetweighted',
    #     # 'interlocus_string_byefo_unweighted_locusweight',
    #     'interlocus_string_byefo_targetweighted_locusweight',
    # ]
    helper = [
        'count_credset_95', 'has_sumstats',
        'gene_count_lte_50k', 'gene_count_lte_100k',
        'gene_count_lte_250k', 'gene_count_lte_500k',
        'proteinAttenuation'
    ]
    dist_foot = ['dist_foot_sentinel_nbh']
    dist_tss = ['dist_tss_sentinel_nbh']

    #
    # Specify feature sets to analyse --------------------------------------------------
    #

    feature_sets = {}

    # Create baseline models
    feature_sets['full_model'] = distance + molecularQTL + interaction + pathogenicity + helper
    # feature_sets['full_model_w_interlocus'] = distance + molecularQTL + interaction + pathogenicity + helper + interlocus

    # Add single distance features
    feature_sets['dist_foot'] = dist_foot
    feature_sets['dist_tss'] = dist_tss

    # Create leave one group out/in (logo/logi) sets
    groups_dict = {
        'distance': distance,
        'molecularQTL': molecularQTL,
        'interaction': interaction,
        'pathogenicity': pathogenicity,
        # 'dist_tss': dist_tss,
        # 'dist_foot': dist_foot,
        # 'interlocus': interlocus,
        # 'helper': helper,
    }
    for grp_name, grp in groups_dict.items():

        # Initate ft sets
        logo = []
        logi = grp + helper

        # Iterate over feature
        for full_ft in feature_sets['full_model']:

            # Find whether the freature is in grp
            if any([full_ft == grp_ft for grp_ft in grp]):
                pass
                # logi.append(full_ft)
            else:
                logo.append(full_ft)

        # Add to feature sets
        feature_sets['logo_{}'.format(grp_name)] = logo
        feature_sets['logi_{}'.format(grp_name)] = logi

    # # Add pairwise between groups
    # for grp_name_A, grp_A in groups_dict.items():
    #     for grp_name_B, grp_B in groups_dict.items():
    #         # Only do for upper triangle
    #         if grp_name_A > grp_name_B:
    #             set_name = 'pairwise_{}_{}'.format(grp_name_A, grp_name_B)
    #             feature_sets[set_name] = grp_A + grp_B + helper

    return feature_sets

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_path', metavar="<file>", help="Input data", type=str, required=True)
    parser.add_argument('--out_dir', metavar="<directory>", help="Output directory for trained models", type=str, required=True)
    # Outcome field
    parser.add_argument('--outcome_field', metavar="<str>", help="Field to use as outcome (default: gold_standard_status)", type=str, default='gold_standard_status')
    # Cross-validation
    parser.add_argument('--cv_outer_folds', metavar="<int>", help="Number of outer folds (default: 5)", type=int, default=5)
    parser.add_argument('--cv_inner_folds', metavar="<int>", help="Number of inner folds (default: 5)", type=int, default=5)
    parser.add_argument('--cv_iters', metavar="<int>", help="Number of iterations for hyperparameter tuning (default: 500)", type=int, default=500)
    # XGBoost paramters
    parser.add_argument('--xgb_early_stop', metavar="<int>", help="Early stopping rounds (default: 10)", type=int, default=10)
    parser.add_argument('--xgb_eval_metric', metavar="<int>", help="Evalution metric for early stopping rounds (default: logloss)", type=str, default='logloss')
    # Other
    parser.add_argument('--random_state', metavar="<int>", help="Random state (default: 123)", type=int, default=123)
    parser.add_argument('--cores', metavar="<int>", help="Number of cores to use. Set as -1 to use all. (default: -1)", type=int, default=-1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
