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
import argparse
import pandas as pd
import joblib
from pprint import pprint
from glob import glob
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *

def main():

    pd.options.mode.chained_assignment = None

    # Parse args
    global args
    args = parse_args()
    studies_to_remove = [
        'GCST007236' # Michailidou K (2015)
    ]

    # Load
    data = pd.read_parquet(args.in_ft)

    # Make spark session
    global spark
    spark = (pyspark.sql.SparkSession.builder.getOrCreate())
    print('Spark version: ', spark.version)

    #
    # Make predictions --------------------------------------------------------
    #

    # Recode True/False
    data = data.replace({True: 1, False: 0})

    # Create output folder
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

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
    pred_df = pd.concat(predictions, ignore_index=True)

    # Convert to spark df
    predictions_long = (
        spark.createDataFrame(pred_df)
        .filter(~col('study_id').isin(*args.exclude_studies))
    )

    #
    # Pivot predictions ------------------------------------------------------
    #

    # Pivot y_proba for each feature set
    group_cols = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id',
                  'training_clf', 'training_gs', 'training_fold']
    predictions_wide = (
        predictions
        .groupby(group_cols)
        .pivot('training_ft')
        .sum('y_proba')
    )

    # Rename pivoted columns
    for coln in [x for x in predictions_wide.columns if x not in group_cols]:
        predictions_wide = predictions_wide.withColumnRenamed(
            coln,
            'y_proba_' + coln
        )

    #
    # Write predictions -------------------------------------------------------
    #

    key_cols = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']

    # Wide format
    (
        predictions_wide
        .repartitionByRange(*key_cols)
        .write
        .parquet(
            args.out_wide,
            mode='overwrite'
        )
    )

    # Long format
    (
        predictions_long
        .repartitionByRange(*key_cols)
        .write
        .parquet(
            args.out_long,
            mode='overwrite'
        )
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_ft', metavar="<parquet>", help="Input feature matrix with gold-standards", type=str, required=True)
    parser.add_argument('--in_model_pattern', metavar="<str>", help="Glob pattern for trained models", type=str, required=True)
    # Params
    parser.add_argument('--exclude_studies', metavar="<str>", help="List of study IDs to exclude", type=str, nargs='+', default=[])
    # Outputs
    parser.add_argument('--out_wide', metavar="<parquet>", help="Output wide parquet containing predictions", type=str, required=True)
    parser.add_argument('--out_long', metavar="<parquet>", help="Output long parquet containing predictions", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
