#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Processes the predictions, features and gold-standards, ready for analysis
but not going to join them here

# Set SPARK_HOME and PYTHONPATH to use 2.4.0
export PYSPARK_SUBMIT_ARGS="--driver-memory 8g pyspark-shell"
export SPARK_HOME=/Users/em21/software/spark-2.4.0-bin-hadoop2.7
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-2.4.0-src.zip:$PYTHONPATH
'''

import os
import sys
from glob import glob
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Window
import joblib
from functools import partial

def main():

    # File paths
    in_pred = 'output/temp/predictions_191111.parquet'
    in_raw_ft = '../../gold_standards/join_gold_standards_to_features/output_raw/features_with_gold_standards.full.191108.parquet'
    in_model_pattern = '../sklearn/output/models_191111/xgboost-full_model-high_medium-*.model.joblib.gz'
    in_studies = '../../features/inputs/191106/studies.parquet'
    in_genes = '../../features/inputs/191106/genes.parquet'
    in_efo_labels = 'extra_datasets/therapeutic_areas.190704.json'
    out_pattern = 'output/{}.191111.parquet'
    studies_to_remove = [
        'GCST007236' # Michailidou K (2015)
    ]

    # Make spark session
    global spark
    spark = (pyspark.sql.SparkSession.builder.getOrCreate())
    print('Spark version: ', spark.version)

    # Make outdir
    os.makedirs(os.path.dirname(out_pattern), exist_ok=True)

    #
    # Load --------------------------------------------------------------------
    #

    key_cols = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']

    # Load features used and per chromosome feature importances
    ft_cols, ft_importances = load_feature_importances_from_models(
        glob(in_model_pattern)
    )

    # Predictions
    predictions = (
        spark.read.parquet(in_pred)
        # .filter(col('study_id') == 'GCST004988') # DEBUG, only do Michailidou K (2017)
    )

    # Gold-standards, dropping features
    gs_cols = ['gs_confidence', 'gs_set', 'gold_standard_status']
    gs = (
        spark.read.parquet(in_raw_ft)
        .select(*key_cols + gs_cols)
        .withColumnRenamed('gold_standard_status', 'gs_status')
    )

    # Raw features, pre-imputation
    ft_raw = (
        spark.read.parquet(in_raw_ft)
        .select(*key_cols + ft_cols)
    )

    #
    # Pivot predictions ------------------------------------------------------
    #

    # Pivot y_proba for each feature set
    group_cols = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id',
                  'training_clf', 'training_gs', 'training_fold']
    predictions_piv = (
        predictions
        .groupby(group_cols)
        .pivot('training_ft')
        .sum('y_proba')
    )

    # Rename pivoted columns
    for coln in [x for x in predictions_piv.columns if x not in group_cols]:
        predictions_piv = predictions_piv.withColumnRenamed(
            coln,
            'y_proba_' + coln
        )

    #
    # Write outputs -----------------------------------------------------------
    #

    # Predictions
    (
        predictions_piv
        .filter(~col('study_id').isin(*studies_to_remove))
        .repartitionByRange(*key_cols)
        .write
        .parquet(
            out_pattern.format('predictions'),
            mode='overwrite'
        )
    )

    # Gold-standards
    (
        gs
        .filter(~col('study_id').isin(*studies_to_remove))
        .repartitionByRange(*key_cols)
        .write
        .parquet(
            out_pattern.format('gold_standards'),
            mode='overwrite'
        )
    )

    # Features: raw
    (
        ft_raw
        .filter(~col('study_id').isin(*studies_to_remove))
        .repartitionByRange(*key_cols)
        .write
        .parquet(
            out_pattern.format('features_raw'),
            mode='overwrite'
        )
    )


    #
    # Create efo and gene label tables ----------------------------------------
    #

    # Make efo label table
    study = (
        spark.read.parquet(in_studies)
        .select('study_id', 'trait_efos')
        .withColumn('efo_code', explode(col('trait_efos')))
        .drop('trait_efos')
    )
    efo_labels = (
        spark.read.json(in_efo_labels)
        .select('efo_code', 'efo_label')
    )
    study = (
        study.join(efo_labels, on='efo_code', how='inner')
        .groupby('study_id')
        .agg(
            collect_list(col('efo_code')).alias('efo_codes'),
            collect_list(col('efo_label')).alias('efo_labels')
        )
        .write
        .parquet(
            out_pattern.format('study2efo'),
            mode='overwrite'
        )
    )

    # Gene table
    (
        spark.read.parquet(in_genes)
        .select('gene_id', 'gene_name')
        .write
        .parquet(
            out_pattern.format('gene_names'),
            mode='overwrite'
        )
    )


    return 0

def weight_features(df, weight_dict, ft_cols):
    ''' Weights each column in ft_cols by weight from weight_dict, 
        specific to the chromosome
    '''
    chrom_weights = weight_dict[df.chrom.values[0]]
    for coln in ft_cols:
        df[coln] = df[coln] * chrom_weights[coln]
    return df

def load_feature_importances_from_models(in_models):
    ''' Loads a list of used features, and a dictionary of feature importances
        form the sklearn model outpus
    Params:
        in_pattern (list of paths): paths to models
    Returns:
        (list of features, dict {chrom: {ft: importance}})
    '''

    ft_cols = None
    ft_importances = {}

    # Iterate over models
    for in_model in in_models:
        
        # Load model
        model = joblib.load(in_model)

        # Get feature column names
        if ft_cols is None:
            ft_cols = model['run_info']['features']
        
        # Get feature importances
        imp_temp = model['model'].best_estimator_.feature_importances_.tolist()

        # Add these to the dict by chromosome
        for chrom in model['run_info']['fold_test_chroms']:
            ft_importances[chrom] = {}
            for ft_name, imp in zip(ft_cols, imp_temp):
                ft_importances[chrom][ft_name] = imp
    
    return ft_cols, ft_importances


if __name__ == '__main__':

    main()
