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
    # in_path = '../inputs/{v}/'.format(v=args.version)
    # out_path = '../output/{v}/separate/'.format(v=args.version)
    max_dist = 500000
    gene_count_bins = [50000, 100000, 250000, 500000]

    # Paths
    in_toploci = in_path + 'toploci.parquet'
    in_credset_v2d = in_path + 'credsets_v2d.parquet'
    in_genes = in_path + 'genes.parquet'
    out_features = out_path + 'dist.parquet'

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

    toploci = (
        spark.read.parquet(in_toploci)
        .drop('neglog_p')
    )
    genes = (
        spark.read.parquet(in_genes)
        .select('gene_id', 'chrom', 'start', 'end', 'tss')
    )

    #
    # Calculate distance from sentinel variant to genes -----------------------
    #

    # Join top loci to genes using non-equi join
    data = toploci.alias('toploci').join(
        genes.alias('genes'),
        (
            (col('toploci.chrom') == col('genes.chrom')) &
            (
                (abs(col('toploci.pos') - col('genes.start')) <= max_dist) |
                (abs(col('toploci.pos') - col('genes.end')) <= max_dist)
            )
        )
    ).drop(col('genes.chrom'))

    # Calculate footprint and tss distance
    data = (
        data
        .withColumn('in_footprint',
            (col('pos') >= col('start')) & (col('pos') <= col('end'))
        )
        .withColumn('start_dist', abs(col('pos')-col('start')))
        .withColumn('end_dist', abs(col('pos')-col('end')))
        .withColumn('min_dist',
            when(
                col('start_dist') <= col('end_dist'),
                col('start_dist')
            ).otherwise(col('end_dist'))
        )
        .withColumn('foot_dist',
            when(
                col('in_footprint'),
                0
            ).otherwise(col('min_dist'))
        )
        .withColumn('tss_dist', abs(col('pos') - col('tss')))
    )

    # Take log of distances
    data = (
        data
        .withColumn('dist_foot_sentinel', log(col('foot_dist') + 1))
        .withColumn('dist_tss_sentinel', log(col('tss_dist') + 1))
    ).cache()

    # Take min grouping by study, locus agg across genes
    dist_sentinel_nbh = (
        data
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(
            min(col('dist_foot_sentinel')).alias('dist_foot_sentinel_nbh_temp'),
            min(col('dist_tss_sentinel')).alias('dist_tss_sentinel_nbh_temp')
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood (in log space)
    data = (
        data.join(
            dist_sentinel_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('dist_foot_sentinel_nbh',
            col('dist_foot_sentinel_nbh_temp') - col('dist_foot_sentinel')
        )
        .drop('dist_foot_sentinel_nbh_temp')
        .withColumn('dist_tss_sentinel_nbh',
            col('dist_tss_sentinel_nbh_temp') - col('dist_tss_sentinel')
        )
        .drop('dist_tss_sentinel_nbh_temp')
    ).cache()

    # Keep sentinel variant features
    sentinel = (
        data
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            'gene_id',
            'dist_foot_sentinel',
            'dist_tss_sentinel',
            'dist_foot_sentinel_nbh',
            'dist_tss_sentinel_nbh'
        )
    )

    #
    # Count number of genes within certain distances of the sentinel ----------
    #

    # Make gene counts based on gene_count_bins
    gene_counts = (
        data
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(*[
            count(when(col('foot_dist') <= count_bin, 1))
            .alias('gene_count_lte_{}k'.format(int(count_bin / 1000)))
            for count_bin in gene_count_bins
        ])
    )

    #
    # Calculate distances over tag variants ------------------------------
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
            'combined_postprob',
            'combined_is95'
        )
        .dropna()
        # .limit(1000) # DEBUG
    )

    # Join tag variants to genes using non-equi join
    data = v2d.alias('v2d').join(
        genes.alias('genes'),
        (
            (col('v2d.tag_chrom') == col('genes.chrom')) &
            (
                (abs(col('v2d.tag_pos') - col('genes.start')) <= max_dist) |
                (abs(col('v2d.tag_pos') - col('genes.end')) <= max_dist)
            )
        )
    ).drop(col('genes.chrom'))

    # Calculate footprint distance
    data = (
        data
        .withColumn('in_footprint',
            (col('tag_pos') >= col('start')) & (col('tag_pos') <= col('end'))
        )
        .withColumn('start_dist', abs(col('tag_pos')-col('start')))
        .withColumn('end_dist', abs(col('tag_pos')-col('end')))
        .withColumn('min_dist',
            when(
                col('start_dist') <= col('end_dist'),
                col('start_dist')
            ).otherwise(col('end_dist'))
        )
        .withColumn('foot_dist',
            when(
                col('in_footprint'),
                0
            ).otherwise(col('min_dist'))
        )
        .withColumn('tss_dist', abs(col('tag_pos') - col('tss')))
    )

    # Take log of distances
    data = (
        data
        .withColumn('log_foot_dist', log(col('foot_dist') + 1))
        .withColumn('log_tss_dist', log(col('tss_dist') + 1))
    ).cache()

    #
    # Calc dist taking min across tags ----------------------------------------
    #

    # Take min grouping by study, locus, gene
    dist_min_local = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .agg(
            min(col('log_foot_dist')).alias('dist_foot_min'),
            min(col('log_tss_dist')).alias('dist_tss_min')
        )
    )

    # Take min grouping by study, locus agg across genes
    dist_min_nbh = (
        data
        .filter(col('combined_is95'))
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(
            min(col('log_foot_dist')).alias('dist_foot_min_nbh_temp'),
            min(col('log_tss_dist')).alias('dist_tss_min_nbh_temp')
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood (in log space)
    dist_min = (
        dist_min_local.join(
            dist_min_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('dist_foot_min_nbh',
            col('dist_foot_min_nbh_temp') - col('dist_foot_min')
        )
        .drop('dist_foot_min_nbh_temp')
        .withColumn('dist_tss_min_nbh',
            col('dist_tss_min_nbh_temp') - col('dist_tss_min')
        )
        .drop('dist_tss_min_nbh_temp')
    )

    #
    # Calc footprint dist taking weighted average across tags -----------------
    #

    # Calculate weighted average distance scores
    dist_ave_scores = (
        data
        # Weight by tag postprob
        .withColumn('foot_dist_weighted',
            col('foot_dist') * col('combined_postprob')
        )
        .withColumn('tss_dist_weighted',
            col('tss_dist') * col('combined_postprob')
        )
        # Take weighted average grouping by study, locus, gene
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .agg(
            log(sum(col('foot_dist_weighted')) + 1).alias('dist_foot_ave'),
            log(sum(col('tss_dist_weighted')) + 1).alias('dist_tss_ave')
        )
    ).cache()

    # Get the min, aggregating over all genes
    dist_ave_nbh = (
        dist_ave_scores
        .groupby('study_id', 'chrom', 'pos', 'ref', 'alt')
        .agg(
            min(col('dist_foot_ave')).alias('dist_foot_ave_nbh_temp'),
            min(col('dist_tss_ave')).alias('dist_tss_ave_nbh_temp')
        )
    )

    # Join local and neighbourhood together, then take proportion of
    # local / neighbourhood (in log space)
    dist_ave = (
        dist_ave_scores.join(
            dist_ave_nbh,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt']
        )
        .withColumn('dist_foot_ave_nbh',
            col('dist_foot_ave_nbh_temp') - col('dist_foot_ave')
        )
        .drop('dist_foot_ave_nbh_temp')
        .withColumn('dist_tss_ave_nbh',
            col('dist_tss_ave_nbh_temp') - col('dist_tss_ave')
        )
        .drop('dist_tss_ave_nbh_temp')
    )


    #
    # Make output -------------------------------------------------------------
    #

    # Join sentinel, min and ave features
    outdata = (
        sentinel.join(
            dist_min,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
            how='outer'
        ).join(
            dist_ave,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
            how='outer'
        ).join(
            gene_counts,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
            how='outer'
        )
    )

    # Only keep rows where dist_foot_sentinel < log(max_dist)
    outdata = outdata.filter(col('dist_foot_sentinel') <= log(lit(max_dist)))

    # Re-order columns
    outdata = outdata.select(
        'study_id',
        'chrom',
        'pos',
        'ref',
        'alt',
        'gene_id',
        'dist_foot_sentinel',
        'dist_foot_sentinel_nbh',
        'dist_foot_min',
        'dist_foot_min_nbh',
        'dist_foot_ave',
        'dist_foot_ave_nbh',
        'dist_tss_sentinel',
        'dist_tss_sentinel_nbh',
        'dist_tss_min',
        'dist_tss_min_nbh',
        'dist_tss_ave',
        'dist_tss_ave_nbh',
        *['gene_count_lte_{}k'.format(int(count_bin / 1000))
          for count_bin in gene_count_bins]
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
