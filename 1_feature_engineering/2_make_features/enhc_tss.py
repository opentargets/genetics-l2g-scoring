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

def main():

    # Parse args
    args = parse_args()
    in_path = 'gs://genetics-portal-dev-staging/l2g/{v}/features/inputs/'.format(v=args.version)
    out_path = 'gs://genetics-portal-dev-staging/l2g/{v}/features/output/separate/'.format(v=args.version)
    
    # Paths
    in_credset_v2d = in_path + 'credsets_v2d.parquet'
    in_v2g = in_path + 'interval.parquet'
    out_features = out_path + 'enhc_tss.parquet'

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

    # Load enhc_tss
    enhc_tss = (
        spark.read.parquet(in_v2g)
        .filter(col('interval_type') == 'fantom5')
        .select(
            col('chrom').alias('tag_chrom'),
            col('pos').alias('tag_pos'),
            col('ref').alias('tag_ref'),
            col('alt').alias('tag_alt'),
            'interval_score',
            'gene_id'
        )
    )

    # Join credsets to tag variants
    data = v2d.join(
        enhc_tss,
        on=['tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt']
    ).cache()

    #
    # Calc max features -------------------------------------------------------
    #

    # Get local max feature
    max_local = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .agg(max(col('interval_score')).alias('enhc_tss_max'))
    )

    # Get neighbourhood max feature
    max_nbh = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(max(col('interval_score')).alias('enhc_tss_max_nbh_temp'))
    )

    # Join max features and take proportion
    max_feats = (
        max_local.join(
            max_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('enhc_tss_max_nbh',
            col('enhc_tss_max') / col('enhc_tss_max_nbh_temp')
        )
        .drop('enhc_tss_max_nbh_temp')
    )

    #
    # Calc ave features -------------------------------------------------------
    #

    # Calculate weighted average scores
    ave_scores = (
        data
        # Weight score column
        .withColumn('enhc_tss_weighted', col('interval_score')*col('combined_postprob'))
        # Take weighted average grouping by study, locus, gene
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .agg(sum(col('enhc_tss_weighted')).alias('enhc_tss_ave'))
    ).cache()

    # Get neighbourhood max
    nbh_max = (
        ave_scores
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(max('enhc_tss_ave').alias('enhc_tss_ave_nbh_temp'))
    )

    # Join and calculate proportion
    ave_out = (
        ave_scores.join(nbh_max,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('enhc_tss_ave_nbh',
            col('enhc_tss_ave') / col('enhc_tss_ave_nbh_temp')
        )
        .drop('enhc_tss_ave_nbh_temp')
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
            mode='ignore'
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
