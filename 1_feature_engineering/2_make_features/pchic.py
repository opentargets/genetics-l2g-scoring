#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#


import os
import sys
import argparse
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def main():

    # Parse args
    args = parse_args()
    in_path = 'gs://genetics-portal-staging/l2g/{v}/features/inputs/'.format(v=args.version)
    out_path = 'gs://genetics-portal-staging/l2g/{v}/features/output/separate/'.format(v=args.version)
    # in_path = '../inputs/{v}/'.format(v=args.version)
    # out_path = '../output/{v}/separate/'.format(v=args.version)
    
    # Paths
    in_credset_v2d = in_path + 'credsets_v2d.parquet'
    in_v2g = in_path + 'interval.parquet'
    out_features = out_path + 'pchic.parquet'

    # Make spark session
    global spark
    spark = (
        pyspark.sql.SparkSession.builder
        .getOrCreate()
    )
    print('Spark version: ', spark.version)

    #
    # Load --------------------------------------------------------------------
    #

    # Load credible set data
    v2d = (
        spark.read.parquet(in_credset_v2d)
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            'tag_chrom',
            'tag_pos',
            'tag_ref',
            'tag_alt',
            'combined_postprob',
            'combined_is95'
        )
        .dropna()
        # .limit(1000) # DEBUG
    )

    # Load pchic
    pchic = (
        spark.read.parquet(in_v2g)
        .filter(col('interval_type') == 'pchic')
        .select(
            col('chrom').alias('tag_chrom'),
            col('pos').alias('tag_pos'),
            col('ref').alias('tag_ref'),
            col('alt').alias('tag_alt'),
            col('interval_feature').alias('tissue'),
            'interval_score',
            'gene_id'
        )
    )

    # Join credsets to tag variants
    data = v2d.join(
        pchic,
        on=['tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt']
    ).cache()

    #
    # Calc max features -------------------------------------------------------
    #

    # Get local max feature
    max_local = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue', 'gene_id')
        .agg(max(col('interval_score')).alias('pchic_max'))
    )

    # Get neighbourhood max feature
    max_nbh = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue')
        .agg(max(col('interval_score')).alias('pchic_max_nbh_temp'))
    )

    # Join max features and take proportion
    max_feats = (
        max_local.join(
            max_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue']
        )
        .withColumn('pchic_max_nbh',
            col('pchic_max') / col('pchic_max_nbh_temp')
        )
        .drop('pchic_max_nbh_temp')
    )

    #
    # Calc ave features -------------------------------------------------------
    #

    # Calculate weighted average scores
    ave_scores = (
        data
        # Weight score column
        .withColumn('pchic_weighted', col('interval_score')*col('combined_postprob'))
        # Take weighted average grouping by study, locus, gene
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue', 'gene_id')
        .agg(sum(col('pchic_weighted')).alias('pchic_ave'))
    ).cache()

    # Get neighbourhood max
    nbh_max = (
        ave_scores
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue')
        .agg(max('pchic_ave').alias('pchic_ave_nbh_temp'))
    )

    # Join and calculate proportion
    ave_out = (
        ave_scores.join(nbh_max,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'tissue']
        )
        .withColumn('pchic_ave_nbh',
            col('pchic_ave') / col('pchic_ave_nbh_temp')
        )
        .drop('pchic_ave_nbh_temp')
    )

    #
    # Take maximum score across tissues ---------------------------------------
    #

    # Take maximum pchic_max across tissues
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .orderBy(desc('pchic_max'), 'tiebreak')
    )
    max_feats = (
        max_feats
        .withColumn('tiebreak', monotonically_increasing_id())
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .drop('rn', 'tiebreak', 'tissue')
    )

    # Take maximum pchic_ave across tissues
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .orderBy(desc('pchic_ave'), 'tiebreak')
    )
    ave_out = (
        ave_out
        .withColumn('tiebreak', monotonically_increasing_id())
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .drop('rn', 'tiebreak', 'tissue')
    )

    #
    # Make output -------------------------------------------------------------
    #

    # Join data
    outdata = max_feats.join(
        ave_out,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
        how='outer'
    )
    
    # Write
    (
        outdata
        .repartitionByRange('study_id', 'chrom', 'pos')
        .write
        .parquet(
            out_features,
            mode='overwrite'
        )
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--version',
        metavar="<str>",
        help="Input data version number",
        type=str,
        required=True
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
