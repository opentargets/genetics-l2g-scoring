#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#

import argparse
import os
import sys
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *
from functools import reduce

def main():

    # Parse args
    args = parse_args()

    # File paths
    in_path = 'gs://genetics-portal-staging/l2g/{v}/features/output/separate/'.format(v=args.version)
    out_path = 'gs://genetics-portal-staging/l2g/{v}/features/output/features.raw.{v}.parquet'.format(v=args.version)

    # Make spark session
    global spark
    spark = (
        pyspark.sql.SparkSession.builder
        .getOrCreate()
    )
    print('Spark version: ', spark.version)

    # -------------------------------------------------------------------------
    # Join all ('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id') dfs
    # except distance
    #

    in_files = [
        'dhs_prmtr.parquet',
        'enhc_tss.parquet',
        'fm_coloc.parquet',
        # 'pchic.parquet',
        'pchicJung.parquet',
        'pics_coloc.parquet',
        'vep.parquet',
        'polyphen.parquet',
        # 'interlocus_string.parquet'
    ]

    # Load list of dfs
    dfs = [
        spark.read.parquet(in_path + inf)
        for inf in in_files
    ]

    # Join based on common columns
    df_full = reduce(
        lambda left, right: left.join(
            right,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
            how='outer'            
        ),
        dfs
    )

    #
    # Join other features -----------------------------------------------------
    #

    # Join credible set counts
    df_full = df_full.join(
        spark.read.parquet(in_path + 'credset95_count.parquet'),
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='left'
    )

    # Join sumstat dummy variable
    df_full = df_full.join(
        spark.read.parquet(in_path + 'sumstat_dummy.parquet'),
        on='study_id',
        how='left'
    )

    # Right join to distance features. Distance is right joined at the end so
    # that we only have protein coding genes within +- 500 kb in the dataset

    df_full = df_full.join(
        spark.read.parquet(in_path + 'dist.parquet'),
        on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
        how='right'
    )

    # Join protein attenuation data
    df_full = df_full.join(
        spark.read.parquet(in_path + 'proteinAttenuation.parquet'),
        on='gene_id',
        how='left'
    )

    #
    # Write output ------------------------------------------------------------
    #

    # Drop any rows that don't have the required keys
    df_full = df_full.dropna(
        subset=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']
    )

    # Drop any rows duplicated rows
    df_full = df_full.drop_duplicates(
        subset=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']
    )

    # Save
    (
        df_full
        .repartitionByRange('study_id', 'chrom', 'pos')
        .sortWithinPartitions('study_id', 'chrom', 'pos', 'ref', 'alt',
                              'gene_id')
        .write
        .parquet(
            out_path,
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
