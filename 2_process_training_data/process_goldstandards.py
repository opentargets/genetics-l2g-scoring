#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#

import argparse
import os
import sys
import pandas as pd
import pyspark.sql
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import Window

def main():

    # Parse args
    global args
    args = parse_args()

    # Make spark session
    global spark
    spark = (pyspark.sql.SparkSession.builder.getOrCreate())
    print('Spark version: ', spark.version)

    #
    # Load --------------------------------------------------------------------
    #

    # Load features
    features = (
        spark.read.parquet(args.in_features)
        .drop('string_partners_bystudy', 'string_partners_byefo')
    )

    # Load gold standards
    gs = (
        spark.read.json(args.in_gs)
        .select(
            col('association_info.otg_id').alias('study_id'),
            col('sentinel_variant.locus_GRCh38.chromosome').alias('chrom'),
            col('sentinel_variant.locus_GRCh38.position').alias('pos'),
            col('sentinel_variant.alleles.reference').alias('ref'),
            col('sentinel_variant.alleles.alternative').alias('alt'),
            col('gold_standard_info.gene_id').alias('gs_gene_id'),
            col('gold_standard_info.highest_confidence').alias('gs_confidence'),
            col('metadata.set_label').alias('gs_set')
        )
        # Filter on confidences
        .filter(col('gs_confidence').isin(*args.gs_classes))
    )

    # Load string and filter string
    string = (
        spark.read.parquet(args.in_string)
        .filter(col('string_score') >= args.min_string_score)
        .filter(col('gene_1_chrom') == col('gene_2_chrom'))
        .select('gene_id_1', 'gene_id_2', 'string_score')
    )

    #
    # Filter gold-standards to keep one per (locus, gene) ---------------------
    #

    # Load cluster information
    clusters = spark.read.parquet(args.in_clusters)

    # Add cluster labels to gold-standards
    gs = gs.join(
        clusters,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='left'
    )

    # Add "unknown" cluster label
    gs = gs.withColumn('cluster_label',
        when(col('cluster_label').isNull(), 'cluster_unknown')
        .otherwise(col('cluster_label'))
    )

    # Join with p-values (used to rank gold-standards at the same locus)
    # This will also drop gold-standards that are not in the top loci table
    toploci = spark.read.parquet(args.in_toploci)
    gs = gs.join(
        toploci,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='left'
    ).cache()

    # Dropout report 1: How many gold-standards have sentinel variant p-values
    # in the OTG data
    (
        gs
        .groupby('gs_set')
        .agg(
            count(when(col('neglog_p').isNotNull(), 1)).alias('count_have_pvals'),
            count(when(col('neglog_p').isNull(), 1)).alias('count_have_no_pvals')
        )
        # Write
        .coalesce(1)
        .write
        .csv(
            os.path.join(args.out_log_dir, '1_in_otg'),
            header=True,
            mode='overwrite'
        )
    )

    # Drop gs not in OTG data
    gs = gs.filter(col('neglog_p').isNotNull())

    # Dropout report 2: Drop "cluster_unknown" cluster labels, these are
    # variants that are not found the 1000 Genomes reference panel, therefore
    # have no LD information available and therefore many features are
    # unavailable. It would not be correct to train the model on these
    # gold-standards
    (
        gs
        .groupby('gs_set')
        .agg(
            count(when(col('cluster_label') != 'cluster_unknown', 1)).alias('count_w_cluster_info'),
            count(when(col('cluster_label') == 'cluster_unknown', 1)).alias('count_wo_cluster_info')
        )
        # Write
        .coalesce(1)
        .write
        .csv(
            os.path.join(args.out_log_dir, '2_has_cluster_information'),
            header=True,
            mode='overwrite'
        )
    )
    gs = gs.filter(col('cluster_label') != 'cluster_unknown')

    # Add column with confidence encoded as High:1, Medium:2, Low:3
    gs = gs.withColumn('gs_confidence_num',
        when(col('gs_confidence') == 'High', 1).otherwise(
            when(col('gs_confidence') == 'Medium', 2).otherwise(
                when(col('gs_confidence') == 'Low', 3)
            )
        )
    )

    # Add column showing rank of gold-standards per (cluster_label, gs_gene_id)
    # Specfiy window spec
    window = (
        Window
        .partitionBy('cluster_label', 'gs_gene_id')
        .orderBy('gs_confidence_num', desc('neglog_p'), 'tiebreak')
    )
    gs = (
        gs
        .withColumn('tiebreak', monotonically_increasing_id())
        .withColumn('gs_rank', rank().over(window))
        .drop('tiebreak')
    ).cache()

    # Dropout report 3: Number of gold-standards after deduplication
    (
        gs
        .groupby('gs_set')
        .agg(
            count(when(col('gs_rank') == 1, 1)).alias('count_kept_post_dedup'),
            count(when(col('gs_rank') > 1, 1)).alias('count_removed_post_dedup')
        )
        # Write
        .coalesce(1)
        .write
        .csv(
            os.path.join(args.out_log_dir, '3_post_deduplication'),
            header=True,
            mode='overwrite'
        )
    )

    # Only keep the top ranked gold-standard per (cluster, gene)
    gs = (
        gs
        .filter(col('gs_rank') == 1)
        .drop('gs_rank')

    )

    #
    # Join gold standards to features -----------------------------------------
    #

    # Merge feature loci to gold-standards
    data = (
        features
        .select('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
        .join(
            gs,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
            how='inner'
        )
    )

    #
    # Join with STRING DB data to find interacting partners -------------------
    #

    # Join string to find interacting partners
    data = (
        data.join(
            string
            .withColumnRenamed('gene_id_1', 'gene_id')
            .withColumnRenamed('gene_id_2', 'gs_gene_id'),
            on=['gene_id', 'gs_gene_id'],
            how='left'
        )
    )

    # Create column stating whether row is:
    # 1:  gold-standard positive
    # 0:  gold-standard negative
    # -1: removed from gold-standard negative due to having interacting partner
    data = (
        data
        .withColumn('gold_standard_status',
            when(col('gene_id') == col('gs_gene_id'), 1).otherwise(
                when(col('string_score').isNotNull(), -1).otherwise(0)
            )
        )
    )

    #
    # Merge gold-standards back onto features ---------------------------------
    #

    # Drop unneeded fields from gold standards
    data = data.drop(
        'gs_gene_id',
        'cluster_label',
        'neglog_p',
        'string_score'
    )

    # Join
    features_w_gs = (
        features.join(
            data,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
            how='left'
        )
    ).cache()

    # Dropout report 4: Total GSP, GSN, removed counts after merging the
    # gold-standards back on the OTG features
    (
        features_w_gs
        .groupby('gs_set')
        .agg(
            count(when(col('gold_standard_status') == 1, 1)).alias('count_GSP'),
            count(when(col('gold_standard_status') == 0, 1)).alias('count_GSN'),
            count(when(col('gold_standard_status') == -1, 1)).alias('count_removed_interaction'),
            count(when(col('gold_standard_status').isNull(), 1)).alias('count_to_predict'),
        )
        # Write
        .coalesce(1)
        .write
        .csv(
            os.path.join(args.out_log_dir, '4_counts_in_OTG_features'),
            header=True,
            mode='overwrite'
        )
    )

    #
    # Write -------------------------------------------------------------------
    #

    # Drop duplicates, there shouldn't be any anyway
    features_w_gs = features_w_gs.drop_duplicates(
        subset=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']
    )

    # Repartition
    features_w_gs = (
        features_w_gs
        .repartitionByRange('study_id', 'chrom', 'pos')
        .cache()
    )

    # Write full data
    (
        features_w_gs
        .write
        .parquet(
            args.out_full,
            mode='overwrite'
        )
    )

    # Write training data only
    (
        features_w_gs
        .filter(col('gold_standard_status').isin(0, 1))
        .write
        .parquet(
            args.out_training,
            mode='overwrite'
        )
    )

    return 0

def parse_args():
    """ Load command line args """
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--in_features', metavar="<file>", help="Input feature matrix", type=str, required=True)
    parser.add_argument('--in_gs', metavar="<file>", help="Input gold-standards", type=str, required=True)
    parser.add_argument('--in_string', metavar="<file>", help="Input StringDB parquet", type=str, required=True)
    parser.add_argument('--in_clusters', metavar="<file>", help="Input cluster information", type=str, required=True)
    parser.add_argument('--in_toploci', metavar="<file>", help="Input toploci information", type=str, required=True)
    # Output paths
    parser.add_argument('--out_full', metavar="<file>", help="Output (full) feature and gold-standard matrix", type=str, required=True)
    parser.add_argument('--out_training', metavar="<file>", help="Output (training only) feature and gold-standard matrix", type=str, required=True)
    parser.add_argument('--out_log_dir', metavar="<directory>", help="Output directory for log information", type=str, required=True)
    # Parameters
    parser.add_argument('--min_string_score', metavar="<file>", help="Minimum stringDB score to be considered a functional partner (default: 0.9)", type=float, default=0.9)
    parser.add_argument('--gs_classes', metavar="<list>", help="List of gold-standard classes to keep (default: [High, Medium, Low])", type=str, nargs='+', default=['High', 'Medium', 'Low'])

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    main()
    