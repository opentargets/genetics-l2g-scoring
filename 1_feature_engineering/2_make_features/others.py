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
    in_path = 'gs://genetics-portal-staging/l2g/{v}/features/inputs/'.format(v=args.version)
    out_path = 'gs://genetics-portal-staging/l2g/{v}/features/output/separate/'.format(v=args.version)

    # Make spark session
    global spark
    spark = (
        pyspark.sql.SparkSession.builder
        .getOrCreate()
    )
    print('Spark version: ', spark.version)

    # Make 95% credible set counts
    in_credset_v2d = in_path + 'credsets_v2d.parquet'
    out_credset_count = out_path + 'credset95_count.parquet'
    (   
        # Load
        spark.read.parquet(in_credset_v2d)
        .filter(col('combined_is95'))
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
        )
        # Count credsets
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(count(lit(1)).alias('count_credset_95'))
        # Write
        .repartitionByRange('study_id', 'chrom', 'pos')
        .write
        .parquet(
            out_credset_count,
            mode='overwrite'
        )
    )
    
    # Make dummy variable showing whether sumstats are available
    in_study = in_path + 'studies.parquet'
    out_sumstat_dummy = out_path + 'sumstat_dummy.parquet'
    (   
        # Load
        spark.read.parquet(in_study)
        .select(
            'study_id',
            'has_sumstats'
        )
        # Write
        .write
        .parquet(
            out_sumstat_dummy,
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
