#!/usr/bin/env python
'''
Preprocesses input data necessary to build feature matrix.
'''
from datetime import datetime
import logging

from hydra import main
from omegaconf import DictConfig
import pyspark.sql.functions as F

from src.utils import *

@main(config_path=f'{get_cwd()}/config', config_name="config")
def main(cfg: DictConfig) -> None:

    today = datetime.now().strftime('%Y-%m-%d')

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
    process_credset_qtl(
        credset_path=cfg.feature_extraction.input.credset,
        posteriorprob_threshold=cfg.feature_extraction.parameters.credset_posteriorprob_threshold,
        out_path=cfg.feature_extraction.processed_inputs.credset_qtl
    )


def process_coloc_table(coloc_path: str, genes_path: str, out_path: str) -> None:
    '''
    Processes coloc table to extract colocalization of molQTLs.

    Inputs:
    - coloc_path: path to directory of parquet files containing coloc data
    - genes_path: path to directory of parquet files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    '''

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
    '''
    Processes gene table to extract protein coding genes.

    Inputs:
    - genes_path: path to directory of parquet files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    '''

    genes = (
        spark.read.parquet(genes_path)
        # Keep only protein-coding genes
        .filter(F.col('biotype') == 'protein_coding').select(
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

    return None


def process_credset_qtl(credset_path: str, posteriorprob_threshold: float, out_path: str) -> None:
    """
    Extract QTL credible set info

    Args:
        in_path (json): credible set results from fine-mapping pipeline
        out_path (parquet)
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
