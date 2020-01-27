#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Predicts values for full data
'''

import sys
import os
import json
import pandas as pd
from pprint import pprint
import joblib
from glob import glob

def main():

    pd.options.mode.chained_assignment = None

    # Args
    in_data = '../../gold_standards/join_gold_standards_to_features/output_raw/features_with_gold_standards.full.191108.parquet'
    in_model_pattern = '../sklearn/output/models_191111/xgboost-*-high_medium-*.model.joblib.gz'
    out_path = 'output/temp/predictions_191111.parquet'

    # Load
    data = pd.read_parquet(in_data)

    # Recode True/False
    data = data.replace({True: 1, False: 0})

    # Create output folder
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load all models and make predictions on corresponding chromosomes
    predictions = []
    model_files = glob(in_model_pattern)
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
    pred_df = pd.concat(predictions, ignore_index=True)

    #
    # Output predictions ------------------------------------------------------
    #

    # Write in long format
    pred_df.to_parquet(out_path)

    return 0


    

if __name__ == '__main__':

    main()
