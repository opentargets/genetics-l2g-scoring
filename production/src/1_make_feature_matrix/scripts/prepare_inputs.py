#!/usr/bin/env python
"""
Preprocesses input data necessary to build feature matrix.
"""
import logging

from hydra import main
from omegaconf import DictConfig
import pyspark.sql.functions as F

from src.utils import *


@main(config_path=f'{get_cwd()}/config', config_name='config')
def main(cfg: DictConfig) -> None:

    # Prepare inputs
    process_coloc_table(
        coloc_path=cfg.feature_extraction.input.coloc,
        genes_path=cfg.feature_extraction.input.genes,
        out_path=cfg.feature_extraction.processed_inputs.coloc,
    )
    process_gene_table(
        genes_path=cfg.feature_extraction.input.genes,
        out_path=cfg.feature_extraction.processed_inputs.genes,
    )
    process_credset_v2d(
        v2d_path=cfg.feature_extraction.input.v2d,
        posteriorprob_threshold=cfg.feature_extraction.parameters.credset_posteriorprob_threshold,
        out_path=cfg.feature_extraction.processed_inputs.credset_qtl,


    )
    process_credset_qtl(
        credset_path=cfg.feature_extraction.input.credset,
        posteriorprob_threshold=cfg.feature_extraction.parameters.credset_posteriorprob_threshold,
        out_path=cfg.feature_extraction.processed_inputs.credset_qtl,
    )


def process_coloc_table(coloc_path: str, genes_path: str, out_path: str) -> None:
    """
    Processes coloc table to extract colocalization of molQTLs.

    Inputs:
    - coloc_path: path to directory of parquet files containing coloc data
    - genes_path: path to directory of parquet files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    """

    coloc = (
        spark.read.parquet(coloc_path).select(
            'left_study',
            'left_chrom',
            'left_pos',
            'left_ref',
            'left_alt',
            'right_type',
            'right_study',
            'right_bio_feature',
            'right_phenotype',
            'right_chrom',
            'right_pos',
            'right_ref',
            'right_alt',
            F.col('right_gene_id').alias('gene_id'),
            'coloc_log2_h4_h3',
        )
        # Keep only tests from molecular QTL datasets
        .filter(F.col('right_type') != 'gwas')
        # Filter on tests with a minimum of 250 intersecting variants
        .filter(F.col('coloc_n_vars') >= 250)
        # Keep only protein-coding genes by inner joining with gene index
        .join(spark.read.parquet(genes_path).select('gene_id'), on='gene_id', how='inner')
        # Repartition
        .repartitionByRange(
            'left_study',
            'left_chrom',
            'left_pos',
        )
    )

    write_parquet(coloc, out_path)
    logging.info(f'Coloc table has been processed and saved to: {out_path}.')

    return None


def process_gene_table(genes_path: str, out_path: str) -> None:
    """
    Processes gene table to extract protein coding genes.

    Inputs:
    - genes_path: path to directory of parquet files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    """

    genes = (
        spark.read.parquet(genes_path)
        # Keep only protein-coding genes
        .filter(F.col('biotype') == 'protein_coding')
        .select(
            'gene_id',
            'gene_name',
            F.col('description').alias('gene_description'),
            F.col('chr').alias('chrom'),
            'start',
            'end',
            'tss',
        )
        .repartition(10)
    )

    write_parquet(genes, out_path)
    logging.info(f'Gene table has been processed and saved to: {out_path}.')


def process_gwas_credset(v2d_path: str, credset_path: str, posteriorprob_threshold: float, out_path: str) -> None:
    """
    It takes the outputs from the fine mapping and LD expansion pipelines to combine them into a single GWAS credible set
    
    Args:
      v2d_path (str): path to the V2D dataset
      credset_path (str): the path to the FM credset table
      posteriorprob_threshold (float): The minimum posterior probability for a variant to be included in
    the credible set.
      out_path (str): path to directory of parquet files containing processed data
    """

    fm = (
        spark.read.parquet(credset_path)
        .filter(F.col('type') == 'gwas')
        .filter(F.col('posteriorprob') >= posteriorprob_threshold)
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
            F.col('postprob').alias('fm_postprob'),
            F.col('is95_credset').alias('fm_is95'),
            F.lit(True).alias('has_fm'),
        )
        .distinct()
    )

    pics = (
        spark.read.parquet(v2d_path)
        .filter(F.col('posterior_prob') >= posteriorprob_threshold)
        .withColumn('has_fm', F.when(F.col('has_sumstats') == True, True).otherwise(False))
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
            F.col('posterior_prob').alias('pics_postprob'),
            # ATTENTION: This field is still not available in the V2D dataset (see #2695)
            F.col('pics_95perc_credset').alias('pics_is95'),
            'has_fm',
        )
        .distinct()
    )

    # Combine the two datasets by joining on the study/tag/lead variants
    joinin_cols = [
        'study_id',
        'lead_chrom',
        'lead_pos',
        'lead_ref',
        'lead_alt',
        'tag_chrom',
        'tag_pos',
        'tag_ref',
        'tag_alt',
    ]
    combined_credset = (
        fm.join(pics, on=joinin_cols, how='outer')
        .withColumn(
            'combined_postprob', F.when(F.col('has_fm') == True, F.col('fm_postprob')).otherwise(F.col('pics_postprob'))
        )
        .withColumn('combined_is95', F.when(F.col('has_fm') == True, F.col('fm_is95')).otherwise(F.col('pics_is95')))
        .drop('has_fm')
        .repartitionByRange('study_id', 'lead_chrom', 'lead_pos')
    )
    write_parquet(combined_credset, out_path)
    logging.info(f'Combined credible set table has been processed and saved to: {out_path}.')


def process_credset_qtl(credset_path: str, posteriorprob_threshold: float, out_path: str) -> None:
    """
    Extract QTL credible set info

    Args:
        in_path (json): credible set results from the fine-mapping pipeline
        out_path (parquet): path to directory of parquet files containing processed data
    """

    qtl_credset_df = (
        spark.read.parquet(credset_path)
        # Only keep mQTLs with a posterior probability of causality higher than the threshold
        .filter(F.col('type') != 'gwas')
        .filter(F.col('postprob') >= posteriorprob_threshold)
        # Create flag to indicate whether tag is sentinel
        .withColumn(
            'is_sentinel',
            F.when(
                (
                    (F.col('lead_chrom') == F.col('tag_chrom'))
                    & (F.col('lead_pos') == F.col('tag_pos'))
                    & (F.col('lead_ref') == F.col('tag_ref'))
                    & (F.col('lead_alt') == F.col('tag_alt'))
                ),
                F.lit(True),
            ).otherwise(F.lit(False)),
        )
        .withColumn('qtl_neglog_p', -F.log10(F.col('tag_pval_cond')))
        .select(
            'study_id',
            'bio_feature',
            'phenotype_id',
            'lead_chrom',
            'lead_pos',
            'lead_ref',
            'lead_alt',
            'is_sentinel',
            'qtl_neglog_p',
        )
        .repartitionByRange('study_id', 'lead_chrom', 'lead_pos')
    )
    write_parquet(qtl_credset_df, out_path)
    logging.info(f'QTL credible set table has been processed and saved to: {out_path}.')


if __name__ == '__main__':

    # Initialise spark and logger
    logging.basicConfig(level=logging.INFO)
    spark = initialize_sparksession()

    main()
