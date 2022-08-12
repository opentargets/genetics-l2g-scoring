#!/usr/bin/env python
"""
Creates features from the colocalization of mQTLs.
"""
from functools import reduce
import logging

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from src.utils import *


def main(cfg: DictConfig) -> None:

    sumstat_coloc = process_standard_coloc(
        coloc_path=cfg.feature_extraction.processed_inputs.coloc,
        qtl_credset_path=cfg.feature_extraction.processed_inputs.credset_qtl,
    )

    ecaviar_coloc = process_ecaviar_coloc(
        qtl_credset_path=cfg.feature_extraction.processed_inputs.credset_qtl,
        v2d_credset_path=cfg.feature_extraction.processed_inputs.credset_v2d,
    )

    coloc_features = sumstat_coloc.unionByName(ecaviar_coloc, allowMissingColumns=True)

    return coloc_features


def process_standard_coloc(coloc_path: str, qtl_credset_path: str) -> DataFrame:
    """
    Processing of colocalization analyses of GWAS summary statistics based on GCTA-COJO.
    """

    qtl_df = (
        spark.read.parquet(qtl_credset_path)
        .select(
            F.col('study_id').alias('right_study'),
            F.col('bio_feature').alias('right_bio_feature'),
            F.col('phenotype_id').alias('right_phenotype'),
            F.col('lead_chrom').alias('right_chrom'),
            F.col('lead_pos').alias('right_pos'),
            F.col('lead_ref').alias('right_ref'),
            F.col('lead_alt').alias('right_alt'),
            'is_sentinel',
            'qtl_neglog_p',
        )
        .filter(F.col('is_sentinel') == True)
        .drop('is_sentinel')
        .distinct()
    )

    coloc_df = (
        spark.read.parquet(coloc_path)
        .join(
            qtl_df,
            on=[
                'right_study',
                'right_bio_feature',
                'right_phenotype',
                'right_chrom',
                'right_pos',
                'right_ref',
                'right_alt',
            ],
            how='left',
        )
        .select(
            F.col('left_study').alias('study_id'),
            F.col('left_chrom').alias('chrom'),
            F.col('left_pos').alias('pos'),
            F.col('left_ref').alias('ref'),
            F.col('left_alt').alias('alt'),
            'gene_id',
            F.col('right_type').alias('qtl_type'),
            'coloc_log2_h4_h3',
            'qtl_neglog_p',
        )
    )

    # Join eqtl, pqtl, sqtl to make final output features
    eqtl_max_coloc = coloc_df.transform(lambda df: extract_qtl_features(df, qtl_type='eqtl', sumstat=True))
    pqtl_max_coloc = coloc_df.transform(lambda df: extract_qtl_features(df, qtl_type='pqtl', sumstat=False))
    sqtl_max_coloc = coloc_df.transform(lambda df: extract_qtl_features(df, qtl_type='pqtl', sumstat=False))

    coloc_features = [eqtl_max_coloc, pqtl_max_coloc, sqtl_max_coloc]
    coloc_features_df = reduce(
        lambda left, right: left.join(right, on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'], how='outer'),
        coloc_features,
    ).distinct()

    return coloc_features_df


def process_ecaviar_coloc(qtl_credset_path: str, v2d_credset_path: str) -> DataFrame:
    """
    Processes the GWAS and QTL credible sets to analyse if the same variant has been found causal.
    """

    pics = (
        spark.read.parquet(v2d_credset_path)
        .select(
            'study_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'tag_chrom',
            'tag_pos',
            'tag_ref',
            'tag_alt',
        )
        # TODO: Open ticket to include pics_postprob in the V2D credible set. This should be coming from the LD matrix.
        .withColumn('pics_postprob', F.rand())
        .dropna()
    )

    qtl = (
        spark.read.parquet(qtl_credset_path)
        .withColumnRenamed('study_id', 'qtl_study_id')
        .withColumnRenamed('type', 'qtl_type')
        .withColumnRenamed('lead_chrom', 'qtl_lead_chrom')
        .withColumnRenamed('lead_pos', 'qtl_lead_pos')
        .withColumnRenamed('lead_ref', 'qtl_lead_ref')
        .withColumnRenamed('lead_alt', 'qtl_lead_alt')
        .withColumnRenamed('postprob', 'qtl_postprob')
    )

    cclp_agg = extract_clpp(pics, qtl)

    # Join eqtl and pqtl to make final output features
    eqtl_max_cclp = cclp_agg.transform(lambda df: extract_qtl_features(df, qtl_type='eqtl', sumstat=False))
    pqtl_max_coloc = cclp_agg.transform(lambda df: extract_qtl_features(df, qtl_type='pqtl', sumstat=False))

    coloc = eqtl_max_cclp.join(
        pqtl_max_coloc,
        on=['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id'],
        how='outer',
    ).distinct()

    return coloc


def extract_qtl_features(qtl_df: DataFrame, qtl_type: str, sumstat: bool) -> DataFrame:
    """
    Extracts local and neighborhood QTL features.
    When the dataset has summary stats, LLR is calculated, which corresponds with the max of coloc_log2_h4_h3 per window.
    When the datased doesn't have summary stats, CLPP is calculated, which corresponds with the max of log_cclp per window.
    """

    coloc_value_input = 'coloc_log2_h4_h3' if sumstat else 'log_cclp'
    coloc_value_output = f'{qtl_type}_coloc_llr_max' if sumstat else f'{qtl_type}_pics_clpp_max'

    window_local = Window.partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id').orderBy(
        F.col(coloc_value_input).desc()
    )
    window_nbh = Window.partitionBy('study_id', 'chrom', 'pos', 'ref', 'alt').orderBy(F.col(coloc_value_input).desc())
    qtl_df = qtl_df.filter(F.col('qtl_type') == qtl_type)

    max_qtl_local_df = (
        qtl_df.withColumn('row_number', F.row_number().over(window_local))
        .filter(F.col('row_number') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            'gene_id',
            F.col('coloc_log2_h4_h3').alias(coloc_value_output),
            F.col('qtl_neglog_p').alias(f'{coloc_value_output}_neglogp'),
        )
        .distinct()
    )

    max_qtl_nbh_df = (
        qtl_df.withColumn('row_number', F.row_number().over(window_nbh))
        .filter(F.col('row_number') == 1)
        .select(
            'study_id',
            'chrom',
            'pos',
            'ref',
            'alt',
            F.col('coloc_log2_h4_h3').alias(f'{coloc_value_output}_nbh_temp'),
        )
        .distinct()
    )

    max_qtl_df = (
        max_qtl_local_df.join(max_qtl_nbh_df, on=['study_id', 'chrom', 'pos', 'ref', 'alt'])
        .withColumn(
            f'{coloc_value_output}_nbh',
            F.col(coloc_value_output) - F.col(f'{coloc_value_output}_nbh_temp'),
        )
        .drop(f'{coloc_value_output}_nbh_temp')
    )

    return max_qtl_df


def extract_clpp(pics: DataFrame, qtl: DataFrame) -> DataFrame:
    """
    Calculates the eCAVIAR CLP probability by joining the credible sets on tag variants.
    """

    intersect = (
        pics.join(qtl, on=['tag_chrom', 'tag_pos', 'tag_ref', 'tag_alt'], how='inner')
        .withColumn('postprob_product', F.col('pics_postprob') * F.col('qtl_postprob'))
        # Calc CCLP, aggregate over all tag variants
        .groupby(
            'study_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'qtl_type',
            'qtl_study_id',
            'bio_feature',
            'phenotype_id',
            'gene_id',
            'qtl_lead_chrom',
            'qtl_lead_pos',
            'qtl_lead_ref',
            'qtl_lead_alt',
        )
        .agg(F.log(F.sum(F.col('postprob_product'))).alias('log_cclp'))
        # Join the neglop_pval for the sentinel QTL variant back onto results
        .join(
            qtl.filter(F.col('is_sentinel') == True).select(
                'qtl_study_id',
                'bio_feature',
                'phenotype_id',
                'qtl_lead_chrom',
                'qtl_lead_pos',
                'qtl_lead_ref',
                'qtl_lead_alt',
                'qtl_neglog_p',
            ),
            on=[
                'qtl_study_id',
                'bio_feature',
                'phenotype_id',
                'qtl_lead_chrom',
                'qtl_lead_pos',
                'qtl_lead_ref',
                'qtl_lead_alt',
            ],
            how='left',
        )
        .withColumnRenamed('lead_chrom', 'chrom')
        .withColumnRenamed('lead_pos', 'pos')
        .withColumnRenamed('lead_ref', 'ref')
        .withColumnRenamed('lead_alt', 'alt')
    )

    return intersect


if __name__ == '__main__':

    # Initialise spark and logger
    logging.basicConfig(level=logging.INFO)
    spark = initialize_sparksession()

    with initialize_config_dir(
        config_dir=Path.joinpath(get_cwd() / 'config').as_posix(), version_base=None, job_name='process_coloc_features'
    ):
        cfg = compose(config_name='config')

    coloc_features_df = main(cfg)

    write_parquet(
        coloc_features_df.repartitionByRange('study_id', 'chrom', 'pos'),
        cfg.feature_extraction.processed_features.coloc,
    )
