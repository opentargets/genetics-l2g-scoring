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
    # in_path = '../inputs/{v}/'.format(v=args.version)
    # out_path = 'output/{v}/separate/'.format(v=args.version)
    
    # Paths
    in_dist = out_path + 'dist.parquet' # Dist features are prerequisite 
    in_string = in_path + 'string.parquet'
    in_study = in_path + 'studies.parquet'
    in_clusters = in_path + 'clusters.parquet'
    out_features = out_path + 'interlocus_string.parquet'

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

    # Load gene distance features as the starting point. This contains all
    # (study, locus, gene) where gene < 500kb from sentinel varaint
    data = (
        spark.read.parquet(in_dist)
        .select('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id')
    )

    # # DEBUG limit
    # print('\nWarning: limiting rows!!!\n')
    # data = data.limit(10)

    # Add cluster information
    clus = spark.read.parquet(in_clusters)
    data = data.join(
        clus,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt'],
        how='inner'
    )

    # Add efo information
    efo = (
        spark.read.parquet(in_study)
        .select('study_id', 'trait_efos')
    )
    data = data.join(
        efo,
        on='study_id',
        how='inner'
    )

    # Load string data
    string = (
        spark.read.parquet(in_string)
        .select(
            col('gene_id_1').alias('gene_id'),
            col('gene_id_2').alias('target_gene_id'),
            'string_score'
        )
    )

    #
    # Calculate number of loci per study and per efo array --------------------
    #

    # Calc num loci per study
    num_loci_per_study = (
        data
        .select(
            'study_id',
            'cluster_label'
        )
        # Deduplicate as rows are currently duplicated by gene
        .drop_duplicates()
        # Count loci per study
        .groupby('study_id')
        .agg(
            count(col('cluster_label')).alias('num_loci_per_study')
        )
    )

    # Merge back
    data = data.join(
        num_loci_per_study,
        on='study_id'    
    ).cache()

    # Join efo array to loci in order to get the number of loci per efo array
    efo_loci = (

        # Left dataset is a deduplicated list of efo arrays
        data
        .select(
            col('trait_efos').alias('source_trait_efos')
        )
        .drop_duplicates()
        .join(
            # Right dataset is a df of efo arrays and cluster labels
            data
            .select(
                col('trait_efos').alias('target_trait_efos'),
                'cluster_label'
            )
            .drop_duplicates(),
            # Non-equi join on whether the efo arrays intersect
            size(
                array_intersect(
                    col('source_trait_efos'),
                    col('target_trait_efos')
                )
            ) > 0
        )
        .drop('target_trait_efos')
    )

    # Count number of loci per efo array
    num_loci_per_efo = (
        efo_loci
        .drop_duplicates()
        .groupby('source_trait_efos')
        .agg(
            count(col('cluster_label')).alias('num_loci_per_efo')
        )
        .withColumnRenamed('source_trait_efos', 'trait_efos')
    )

    # Merge back
    data = data.join(
        num_loci_per_efo,
        on='trait_efos'    
    ).cache()

    #
    # Calculate study inter-locus scores --------------------------------------
    #

    # Add source_ prefix to columns in data_string
    data_string_source = (
        data_string
        .select(*(col(x).alias('source_' + x) for x in data_string.columns))
    )
    # Add target_ prefix to columns in data
    data_targets = (
        data
        .select(*(col(x).alias('target_' + x) for x in data.columns))
    )

    # Join source and target data in order to find other loci that have string
    # partners, ensuring that the studies match but the loci (cluster labels)
    # are different
    study_inter = (
        data_string_source
        .join(
            data_targets,
            (   
                # The target gene_id matches the source target id
                (col('target_gene_id') == col('source_target_gene_id')) &
                # The studies match
                (col('target_study_id') == col('source_study_id')) &
                # The loci are different
                (col('target_cluster_label') != col('source_cluster_label'))
            ),
            how='inner'
        )
        # Only keep required columns as we have lots of duplicated fields and
        # things are getting messy
        .select(
            col('source_gene_id').alias('gene_id'),
            col('source_study_id').alias('study_id'),
            col('source_chrom').alias('chrom'),
            col('source_pos').alias('pos'),
            col('source_ref').alias('ref'),
            col('source_alt').alias('alt'),
            col('source_num_loci_per_study').alias('num_loci_per_study'),
            col('source_string_score').alias('string_score'),
            col('target_gene_id').alias('target_gene_id'),
            col('target_cluster_label').alias('target_cluster_label')
        )
    ).persist()

    # Calculate target weights, this is 1 / count(partners) as each target locus
    target_weights = (
        study_inter
        .groupby(
            'study_id', 'chrom', 'pos', 'ref', 'alt',
            'gene_id', 'target_cluster_label'
        )
        .agg(
            (1/countDistinct(col('target_gene_id'))).alias('target_weight'),
        )
    )

    # Merge weights back
    study_inter = (
        study_inter
        .join(
            target_weights,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt',
                'gene_id', 'target_cluster_label'],
            how='inner'
        )
    )

    # Then need to drop duplicate target_gene_id, as we don't want any to be
    # counted twice+ if they are found in multiple target loci. Duplicates will
    # be dropped at random as Spark.drop_duplicates() is non determinative.
    study_inter = study_inter.drop_duplicates(
        subset=['study_id', 'chrom', 'pos', 'ref', 'alt',
                'gene_id', 'target_gene_id']
    )

    # Calculate scores
    interlocus_study_scores = (
        study_inter
        .groupby('study_id', 'chrom', 'pos', 'ref',
                 'alt', 'gene_id')
        .agg(
            # Create unweighted score
            sum(col('string_score')).alias('interlocus_string_bystudy_unweighted'),
            # Create weighted scores
            sum(col('string_score') * col('target_weight')).alias('interlocus_string_bystudy_targetweighted'),
            sum(col('string_score') * (1/col('num_loci_per_study'))).alias('interlocus_string_bystudy_unweighted_locusweight'),
            sum(col('string_score') * col('target_weight') * (1/col('num_loci_per_study'))).alias('interlocus_string_bystudy_targetweighted_locusweight'),
            # Create array of string partners
            collect_list(array(col('target_gene_id'), col('string_score'))).alias('string_partners_bystudy')

        )
    )

    
    #
    # Do the same but for EFO level inter-locus scores ---------------------
    #

    # Join source and target data in order to find other loci that have string
    # partners, ensuring that the efo arrays match but the loci (cluster labels)
    # are different
    efo_inter = (
        data_string_source
        .join(
            data_targets,
            (   
                # The target gene_id matches the source target id
                (col('target_gene_id') == col('source_target_gene_id')) &
                # The efo arrays intersect
                (size(array_intersect(col('source_trait_efos'),
                 col('target_trait_efos'))) > 0) &
                # The loci are different
                (col('target_cluster_label') != col('source_cluster_label'))
            ),
            how='inner'
        )
        # Only keep required columns as we have lots of duplicated fields and
        # things are getting messy
        .select(
            col('source_gene_id').alias('gene_id'),
            col('source_study_id').alias('study_id'),
            col('source_chrom').alias('chrom'),
            col('source_pos').alias('pos'),
            col('source_ref').alias('ref'),
            col('source_alt').alias('alt'),
            col('source_num_loci_per_efo').alias('num_loci_per_efo'),
            col('source_string_score').alias('string_score'),
            col('target_gene_id').alias('target_gene_id'),
            col('target_cluster_label').alias('target_cluster_label')
        )
    ).persist()

    # Calculate target weights, this is 1 / count(partners) as each target locus
    target_weights = (
        efo_inter
        .groupby(
            'study_id', 'chrom', 'pos', 'ref', 'alt',
            'gene_id', 'target_cluster_label'
        )
        .agg(
            (1/countDistinct(col('target_gene_id'))).alias('target_weight'),
        )
    )

    # Merge weights back
    efo_inter = (
        efo_inter
        .join(
            target_weights,
            on=['study_id', 'chrom', 'pos', 'ref', 'alt',
                'gene_id', 'target_cluster_label'],
            how='inner'
        )
    )

    # Then need to drop duplicate target_gene_id, as we don't want any to be
    # counted twice+ if they are found in multiple target loci. Duplicates will
    # be dropped at random as Spark.drop_duplicates() is non determinative.
    efo_inter = efo_inter.drop_duplicates(
        subset=['study_id', 'chrom', 'pos', 'ref', 'alt',
                'gene_id', 'target_gene_id']
    )

    # Calculate scores
    interlocus_efo_scores = (
        efo_inter
        .groupby('study_id', 'chrom', 'pos', 'ref',
                 'alt', 'gene_id')
        .agg(
            # Create unweighted score
            sum(col('string_score')).alias('interlocus_string_byefo_unweighted'),
            # Create weighted scores
            sum(col('string_score') * col('target_weight')).alias('interlocus_string_byefo_targetweighted'),
            sum(col('string_score') * (1/col('num_loci_per_efo'))).alias('interlocus_string_byefo_unweighted_locusweight'),
            sum(col('string_score') * col('target_weight') * (1/col('num_loci_per_efo'))).alias('interlocus_string_byefo_targetweighted_locusweight'),
            # Create array of string partners
            collect_list(array(col('target_gene_id'), col('string_score'))).alias('string_partners_byefo')
        )
    )

    #
    # Join the by study and by EFO scores together and write output -----------
    #

    # Join
    outdata = interlocus_study_scores.join(
        interlocus_efo_scores,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id']
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
