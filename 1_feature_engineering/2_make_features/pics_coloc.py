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
    
    # Paths
    in_credset_v2d = in_path + 'credsets_v2d.parquet'
    in_credset_qtl = in_path + 'credsets_qtl.parquet'
    out_features = out_path + 'pics_coloc.parquet'

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

    pics = (
        spark.read.parquet(in_credset_v2d)
        .select('study_id', 'lead_chrom', 'lead_pos', 'lead_ref', 'lead_alt',
                'tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt', 'pics_postprob')
        .dropna()
        # .limit(10000) # DEBUG
    )
    qtl = (
        spark.read.parquet(in_credset_qtl)
        .withColumnRenamed('study_id', 'qtl_study_id')
        .withColumnRenamed('type', 'qtl_type')
        .withColumnRenamed('lead_chrom', 'qtl_lead_chrom')
        .withColumnRenamed('lead_pos', 'qtl_lead_pos')
        .withColumnRenamed('lead_ref', 'qtl_lead_ref')
        .withColumnRenamed('lead_alt', 'qtl_lead_alt')
        .withColumnRenamed('postprob', 'qtl_postprob')
    )

    #
    # Calc eCaviar CCLP -------------------------------------------------------
    #

    # Join on tag variants
    intersect = pics.join(
        qtl,
        on=['tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt']
    )

    # Calc product of the posteriors
    intersect = intersect.withColumn('postprob_product',
        col('pics_postprob') * col('qtl_postprob')
    )

    # Calc CCLP, aggregate over all tag variants
    clpp_agg = (
        intersect
        .groupby(
            'study_id', 'lead_chrom', 'lead_pos', 'lead_ref', 'lead_alt',
            'qtl_type', 'qtl_study_id', 'bio_feature', 'phenotype_id',
            'gene_id', 'qtl_lead_chrom', 'qtl_lead_pos', 'qtl_lead_ref',
            'qtl_lead_alt'
        )
        .agg(
            log(sum(col('postprob_product'))).alias('log_cclp')
        )
    )

    # #
    # # Compare with coloc results ----------------------------------------------
    # #

    # # Merge with coloc results so that they can be compared
    # coloc = (
    #     spark.read.parquet('../inputs/{v}/coloc.parquet'.format(v=args.version))
    #     .select(
    #         col('left_study').alias('study_id'),
    #         col('left_chrom').alias('lead_chrom'),
    #         col('left_pos').alias('lead_pos'),
    #         col('left_ref').alias('lead_ref'),
    #         col('left_alt').alias('lead_alt'),
    #         col('right_type').alias('qtl_type'),
    #         col('right_study').alias('qtl_study_id'),
    #         col('right_bio_feature').alias('bio_feature'),
    #         col('right_phenotype').alias('phenotype_id'),
    #         col('right_chrom').alias('qtl_lead_chrom'),
    #         col('right_pos').alias('qtl_lead_pos'),
    #         col('right_ref').alias('qtl_lead_ref'),
    #         col('right_alt').alias('qtl_lead_alt'),
    #         'gene_id',
    #         'coloc_log2_h4_h3',
    #         'coloc_h4'
    #     )
    # )
    # coloc_join = clpp_agg.join(
    #     coloc,
    #     on=['study_id', 'lead_chrom', 'lead_pos', 'lead_ref', 'lead_alt',
    #         'qtl_type', 'qtl_study_id', 'bio_feature', 'phenotype_id',
    #         'gene_id', 'qtl_lead_chrom', 'qtl_lead_pos', 'qtl_lead_ref',
    #         'qtl_lead_alt']
    # )

    # # Write output for inspection
    # (
    #     coloc_join
    #     .write.csv(
    #         out_temp_coloc,
    #         header=True,
    #         mode='ignore'
    #     )
    # )

    #
    # Join the neglop_pval for the sentinel QTL variant back onto results -----
    #

    qtl_pvals = (
        qtl
        .filter(col('is_sentinel'))
        .select('qtl_study_id', 'bio_feature', 'phenotype_id', 'qtl_lead_chrom',
                'qtl_lead_pos', 'qtl_lead_ref', 'qtl_lead_alt', 'qtl_neglog_p')
    )
    clpp_agg = clpp_agg.join(
        qtl_pvals,
        on=['qtl_study_id', 'bio_feature', 'phenotype_id', 'qtl_lead_chrom',
                'qtl_lead_pos', 'qtl_lead_ref', 'qtl_lead_alt'],
        how='left'
    )

    clpp_agg = clpp_agg.cache()

    #
    # Extract eQTL features ---------------------------------------------------
    #

    clpp_agg_eqtl = clpp_agg.filter(col('qtl_type') == 'eqtl')

    # Get log_cclp, qtl_neglog_p from row with max(log_cclp)
    window_spec = (
        Window
        .partitionBy('study_id', 'lead_chrom', 'lead_pos', 'lead_ref',
                     'lead_alt', 'gene_id')
        .orderBy(desc('log_cclp'))
    )
    eqtl_max_cclp_local = (
        clpp_agg_eqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            'gene_id',
            col('log_cclp').alias('eqtl_pics_clpp_max'),
            col('qtl_neglog_p').alias('eqtl_pics_clpp_max_neglogp')
        )
    )

    # Get log_cclp from row with max(log_cclp) across any gene at each locus
    # (neighbourhood max)
    window_spec = (
        Window
        .partitionBy('study_id', 'lead_chrom', 'lead_pos', 'lead_ref',
                     'lead_alt')
        .orderBy(desc('log_cclp'))
    )
    eqtl_max_cclp_nbh = (
        clpp_agg_eqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            col('log_cclp').alias('eqtl_pics_clpp_max_nhb_temp')
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood
    eqtl_max_cclp = (
        eqtl_max_cclp_local.join(
            eqtl_max_cclp_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('eqtl_pics_clpp_max_nhb',
            col('eqtl_pics_clpp_max') - col('eqtl_pics_clpp_max_nhb_temp')
            
        )
        .drop('eqtl_pics_clpp_max_nhb_temp')
    )

    #
    # Extract pQTL features ---------------------------------------------------
    #

    clpp_agg_pqtl = clpp_agg.filter(col('qtl_type') == 'pqtl')

    # Get log_cclp, qtl_neglog_p from row with max(log_cclp)
    window_spec = (
        Window
        .partitionBy('study_id', 'lead_chrom', 'lead_pos', 'lead_ref',
                     'lead_alt', 'gene_id')
        .orderBy(desc('log_cclp'))
    )
    pqtl_max_cclp_local = (
        clpp_agg_pqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            'gene_id',
            col('log_cclp').alias('pqtl_pics_clpp_max'),
            col('qtl_neglog_p').alias('pqtl_pics_clpp_max_neglogp')
        )
    )

    # Get log_cclp from row with max(log_cclp) across any gene at each locus
    # (neighbourhood max)
    window_spec = (
        Window
        .partitionBy('study_id', 'lead_chrom', 'lead_pos', 'lead_ref',
                     'lead_alt')
        .orderBy(desc('log_cclp'))
    )
    pqtl_max_cclp_nbh = (
        clpp_agg_pqtl
        .withColumn('rn', row_number().over(window_spec))
        .filter(col('rn') == 1)
        .select(
            'study_id',
            col('lead_chrom').alias('chrom'),
            col('lead_pos').alias('pos'),
            col('lead_ref').alias('ref'),
            col('lead_alt').alias('alt'),
            col('log_cclp').alias('pqtl_pics_clpp_max_nhb_temp')
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood
    pqtl_max_cclp = (
        pqtl_max_cclp_local.join(
            pqtl_max_cclp_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('pqtl_pics_clpp_max_nhb',
            col('pqtl_pics_clpp_max') - col('pqtl_pics_clpp_max_nhb_temp')
        )
        .drop('pqtl_pics_clpp_max_nhb_temp')
    )

    #
    # Join eqtl and pqtl to make final output features ------------------------
    #

    data = eqtl_max_cclp.join(
        pqtl_max_cclp,
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
