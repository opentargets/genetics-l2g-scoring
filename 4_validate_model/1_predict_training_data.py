#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Takes trained models and predicts gold-standards
'''

import sys
import os
import json
import pandas as pd
from pprint import pprint
import joblib
from glob import glob
import numpy
import argparse

def main():

    pd.options.mode.chained_assignment = None

    # Parse args
    global args
    args = parse_args()

    # Load
    data = pd.read_parquet(args.in_ft)

    # Recode True/False
    data = data.replace({True: 1, False: 0})    

    # Initiate feature importance results
    feature_imp = {}

    # Load all models and make predictions on corresponding chromosomes
    predictions = []
    model_files = glob(args.in_model_pattern)
    for c, model_file in enumerate(model_files):

        print('Processing model {} of {}...'.format(c, len(model_files)))

        # Load model
        model = joblib.load(model_file)

        # # Print inner and outer scores
        # print(model['run_info']['fold_name'], model['scores'])
        
        #
        # Extract feature importances -----------------------------------------
        #

        # Get keys
        clf = model['run_info']['classifier_name']
        ft = model['run_info']['feature_name']
        gs = model['run_info']['gold_standard_set']
        fn = model['run_info']['fold_name']

        # Create entry in output dict
        if clf not in feature_imp:
            feature_imp[clf] = {}
        if ft not in feature_imp[clf]:
            feature_imp[clf][ft] = {'feature_names': model['run_info']['features']}
        if gs not in feature_imp[clf][ft]:
            feature_imp[clf][ft][gs] = {}
        
        # Add feature importances to output
        feature_imp[clf][ft][gs][fn] = {
            'feature_importances': model['model'].best_estimator_.feature_importances_.tolist()
        }

        #
        # Make predictions ----------------------------------------------------
        #

        # Subset test dataset by chromosomes
        test_data = data.loc[
            data['chrom'].isin(model['run_info']['fold_test_chroms']), :
        ]

        # Make predictions
        test_data['y_pred'] = model['model'].best_estimator_.predict(
            test_data.loc[:, model['run_info']['features']]
        )
        test_data['y_proba'] = model['model'].best_estimator_.predict_proba(
            test_data.loc[:, model['run_info']['features']]
        )[:, 1]

        # Add classifier information as fields to df        
        for key in ['classifier_name', 'feature_name', 'gold_standard_set', 'fold_name']:
            test_data['clf_{}'.format(key)] = model['run_info'][key]
        
        # Create a single classifier key
        test_data['clf_key'] = '-'.join([
            model['run_info']['classifier_name'],
            model['run_info']['feature_name'],
            model['run_info']['gold_standard_set']
        ]).replace(' ', '_')
        
        # Add to list of predictions
        predictions.append(test_data)

        # if c == 3:
        #     break

    # Concatenate all predictions together
    pred_df = pd.concat(predictions, ignore_index=True)

    #
    # Output predictions long format ------------------------------------------------------
    #

    # Write in long format
    os.makedirs(os.path.dirname(args.out_pred), exist_ok=True)
    pred_df.to_parquet(args.out_pred)

    #
    # Write feature importances -----------------------------------------------
    #

    os.makedirs(os.path.dirname(args.out_ftimp), exist_ok=True)
    with open(args.out_ftimp, 'w') as out_h:
        json.dump(feature_imp, out_h, indent=2)
    
    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_ft', metavar="<parquet>", help="Input feature matrix with gold-standards", type=str, required=True)
    parser.add_argument('--in_model_pattern', metavar="<str>", help="Glob pattern for trained models", type=str, required=True)
    # Outputs
    parser.add_argument('--out_pred', metavar="<parquet>", help="Output parquet containing predictions", type=str, required=True)
    parser.add_argument('--out_ftimp', metavar="<json>", help="Output json containing feature importances", type=str, required=True)

    args = parser.parse_args()
    return args
    

if __name__ == '__main__':

    main()
