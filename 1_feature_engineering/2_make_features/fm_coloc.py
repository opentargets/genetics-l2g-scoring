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
    in_path = 'gs://genetics-portal-dev-staging/l2g/{v}/features/inputs/'.format(v=args.version)
    out_path = 'gs://genetics-portal-dev-staging/l2g/{v}/features/output/separate/'.format(v=args.version)
    
    # Paths
    in_coloc = in_path + 'coloc.parquet'
    in_credset_qtl = in_path + 'credsets_qtl.parquet'
    out_features = out_path + 'fm_coloc.parquet'

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

    coloc = (
        spark.read.parquet(in_coloc)
        # .limit(1000) # DEBUG
    )

    # Load neglog_pvalue of sentinel qtl variant
    qtl = (
        spark.read.parquet(in_credset_qtl)
        # .limit(10000) # DEBUG
        .withColumnRenamed('study_id', 'right_study')
        .withColumnRenamed('bio_feature', 'right_bio_feature')
        .withColumnRenamed('phenotype_id', 'right_phenotype')
        .withColumnRenamed('lead_chrom', 'right_chrom')
        .withColumnRenamed('lead_pos', 'right_pos')
        .withColumnRenamed('lead_ref', 'right_ref')
        .withColumnRenamed('lead_alt', 'right_alt')
        .filter(col('is_sentinel'))
        .select('right_study', 'right_bio_feature', 'right_phenotype',
                'right_chrom', 'right_pos', 'right_ref', 'right_alt',
                'qtl_neglog_p')
        .drop_duplicates()
    )
    coloc = coloc.join(
        qtl,
        on=['right_study', 'right_bio_feature', 'right_phenotype',
            'right_chrom', 'right_pos', 'right_ref', 'right_alt'],
        how='left'
    )

    # Drop unneeded columns
    coloc = coloc.select(
        col('left_study').alias('study_id'),
        col('left_chrom').alias('chrom'),
        col('left_pos').alias('pos'),
        col('left_ref').alias('ref'),
        col('left_alt').alias('alt'),
        'gene_id',
        col('right_type').alias('qtl_type'),
        'coloc_log2_h4_h3',
        'qtl_neglog_p'
    ).cache()

    #
    # Extract eQTL features ---------------------------------------------------
    #

    coloc_eqtl = coloc.filter(col('qtl_type') == 'eqtl')

    # Get eqtl_coloc_llr_max, qtl_neglog_p from row with max(eqtl_coloc_llr_max)
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .orderBy(desc('coloc_log2_h4_h3'))
    )
    eqtl_max_coloc_local = (
        coloc_eqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            'gene_id',
            col('coloc_log2_h4_h3').alias('eqtl_coloc_llr_max'),
            col('qtl_neglog_p').alias('eqtl_coloc_llr_max_neglogp')
        )
    )

    # Get eqtl_coloc_llr_max from row with max(eqtl_coloc_llr_max) across any
    # gene at each locus (neighbourhood max)
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt')
        .orderBy(desc('coloc_log2_h4_h3'))
    )
    eqtl_max_coloc_nbh = (
        coloc_eqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            col('coloc_log2_h4_h3').alias('eqtl_coloc_llr_max_nbh_temp'),
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood
    eqtl_max_coloc = (
        eqtl_max_coloc_local.join(
            eqtl_max_coloc_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('eqtl_coloc_llr_max_nbh',
            col('eqtl_coloc_llr_max') - col('eqtl_coloc_llr_max_nbh_temp')
        )
        .drop('eqtl_coloc_llr_max_nbh_temp')
    )

    #
    # Extract pQTL features ---------------------------------------------------
    #

    coloc_pqtl = coloc.filter(col('qtl_type') == 'pqtl')

    # Get pqtl_coloc_llr_max, qtl_neglog_p from row with max(pqtl_coloc_llr_max)
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .orderBy(desc('coloc_log2_h4_h3'))
    )
    pqtl_max_coloc_local = (
        coloc_pqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            'gene_id',
            col('coloc_log2_h4_h3').alias('pqtl_coloc_llr_max'),
            col('qtl_neglog_p').alias('pqtl_coloc_llr_max_neglogp')
        )
    )

    # Get pqtl_coloc_llr_max from row with max(pqtl_coloc_llr_max) across any
    # gene at each locus (neighbourhood max)
    window_spec = (
        Window
        .partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt')
        .orderBy(desc('coloc_log2_h4_h3'))
    )
    pqtl_max_coloc_nbh = (
        coloc_pqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            col('coloc_log2_h4_h3').alias('pqtl_coloc_llr_max_nbh_temp'),
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood
    pqtl_max_coloc = (
        pqtl_max_coloc_local.join(
            pqtl_max_coloc_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('pqtl_coloc_llr_max_nbh',
            col('pqtl_coloc_llr_max') - col('pqtl_coloc_llr_max_nbh_temp')
        )
        .drop('pqtl_coloc_llr_max_nbh_temp')
    )
    

    #
    # Join eqtl and pqtl to make final output features ------------------------
    #

    data = eqtl_max_coloc.join(
        pqtl_max_coloc,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
        how='outer'
    )

    # Write
    (
        data
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
