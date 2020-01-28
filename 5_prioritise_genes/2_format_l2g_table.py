#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Format L2G table
'''

import sys
import os
import argparse
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *

def main():

    # Parse args
    global args
    args = parse_args()

    # Make spark session
    global spark
    spark = (pyspark.sql.SparkSession.builder.getOrCreate())
    print('Spark version: ', spark.version)

    # Load
    predictions = (
        spark.read.parquet(args.in_long)
        .filter(~col('study_id').isin(*args.exclude_studies))
        .filter(col('training_clf') == args.keep_clf)
        .filter(col('training_gs') == args.keep_gsset)
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

    # Wide format
    (
        predictions_wide
        .repartitionByRange('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .write
        .parquet(
            args.out_l2g,
            mode='overwrite'
        )
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_long', metavar="<parquet>", help="Input feature matrix with gold-standards", type=str, required=True)
    # Params
    parser.add_argument('--exclude_studies', metavar="<str>", help="List of study IDs to exclude", type=str, nargs='+', default=[])
    parser.add_argument('--keep_clf', metavar="<str>", help="Label of classifier to keep (default: xgboost)", type=str, default='xgboost')
    parser.add_argument('--keep_gsset', metavar="<str>", help="Label of gold-standard set to keep (default: high_medium)", type=str, default='high_medium')
    # Outputs
    parser.add_argument('--out_l2g', metavar="<parquet>", help="Output wide parquet containing predictions", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
