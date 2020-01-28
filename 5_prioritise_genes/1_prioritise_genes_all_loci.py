#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Predicts values for all data in feature matrix
'''

import sys
import os
import argparse
import pandas as pd
import joblib
from glob import glob

def main():

    pd.options.mode.chained_assignment = None

    # Parse args
    global args
    args = parse_args()

    # Load
    print('Loading input...')
    data = pd.read_parquet(args.in_ft)

    #
    # Make predictions --------------------------------------------------------
    #

    # Recode True/False
    data = data.replace({True: 1, False: 0})

    # Load all models and make predictions on corresponding chromosomes
    predictions = []
    model_files = glob(args.in_model_pattern)
    for c, model_file in enumerate(model_files):

        # Skip if model is "pairwise"
        if "pairwise_" in model_file:
            continue

        print('Processing model {} of {}...'.format(c, len(model_files)))

        # Load model
        model = joblib.load(model_file)

        #
        # Make predictions ----------------------------------------------------
        #

        # Subset test dataset by chromosomes
        test_data = data.loc[
            data['chrom'].isin(model['run_info']['fold_test_chroms']), :
        ]

        # Make predictions
        test_data['y_pred'] = model['model'].predict(
            test_data.loc[:, model['run_info']['features']]
        )
        test_data['y_proba'] = model['model'].predict_proba(
            test_data.loc[:, model['run_info']['features']]
        )[:, 1]

        # Add classifier information as fields to df        
        for key in ['classifier_name', 'feature_name', 'gold_standard_set', 'fold_name']:
            test_data['clf_{}'.format(key)] = model['run_info'][key]
        
        # Rename
        test_data = test_data.rename(columns={
            'clf_classifier_name': 'training_clf',
            'clf_feature_name': 'training_ft',
            'clf_gold_standard_set': 'training_gs',
            'clf_fold_name': 'training_fold',
        })

        # Drop uneeded columns
        cols_keep = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id',
                    'y_pred', 'y_proba', 'training_clf', 'training_ft',
                    'training_gs', 'training_fold']
        prediction = test_data.loc[:, cols_keep]
        
        # Add to list of predictions
        predictions.append(prediction)

    # Concatenate all predictions together
    print('Concatenating model outputs...')
    pred_df = pd.concat(predictions, ignore_index=True)

    # Write as parquet using pyarrow
    os.makedirs(os.path.dirname(args.out_long), exist_ok=True)
    pred_df.to_parquet(
        args.out_long,
        engine='pyarrow',
        compression='snappy',
        flavor='spark'
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_ft', metavar="<parquet>", help="Input feature matrix with gold-standards", type=str, required=True)
    parser.add_argument('--in_model_pattern', metavar="<str>", help="Glob pattern for trained models", type=str, required=True)
    # Outputs
    parser.add_argument('--out_long', metavar="<parquet>", help="Output long parquet containing predictions", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
