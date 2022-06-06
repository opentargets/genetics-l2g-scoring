#!/usr/bin/env python
'''
Preprocesses input data necessary to build feature matrix.
'''
from datetime import datetime
import logging

from pathlib import Path
from hydra import compose, initialize_config_dir
import pyspark.sql.functions as F

from utils import *


def main(coloc_table: str, genes_table: str, coloc_out_path: str, genes_out_path: str) -> None:

    today = datetime.now().strftime('%Y-%m-%d')

    # Parse output args
    coloc_out_path: str = f"data/coloc-{today}.parquet"
    genes_out_path: str = f"data/genes-{today}.parquet"

    # Prepare inputs
    process_coloc_table(
        coloc_path=coloc_table,
        genes_path=genes_table,
        out_path=coloc_out_path,
    )
    process_gene_table(
        genes_path=genes_table,
        out_path=genes_out_path,
    )


def process_coloc_table(coloc_path: str, genes_path: str, out_path: str) -> None:
    '''
    Processes coloc table to extract colocalization of molQTLs.

    Inputs:
    - coloc_path: path to directory of JSON files containing coloc data
    - genes_path: path to directory of JSON files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    '''

    coloc = (
        spark.read.json(coloc_path).select(
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
        .join(spark.read.json(genes_path).select('gene_id'), on='gene_id', how='inner')
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
    - genes_path: path to directory of JSON files containing gene data

    Outputs:
    - out_path: path to directory of parquet files containing processed data
    '''

    genes = (
        spark.read.json(genes_path)
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
        # Repartition
        .repartition(10)
    )
    
    write_parquet(genes, out_path)
    logging.info(f'Gene table has been processed and saved to: {out_path}.')

    return None

if __name__ == '__main__':

    # Load config
    initialize_config_dir(version_base=None, config_dir=Path.joinpath(get_cwd() / 'conf').as_posix(), job_name='prepare_fm_inputs')
    cfg = compose(config_name='config')

    # Initialise spark and logger
    logging.basicConfig(level=logging.INFO)
    spark = initialize_sparksession()

    today = datetime.now().strftime('%Y-%m-%d')

    print(f"{cfg.intermediate.coloc}-{today}")
'''
    main(
        coloc_table=cfg.inputs.coloc,
        genes_table=cfg.inputs.genes,
        coloc_out_path=f"{cfg.intermediate.coloc}-{today}",
        genes_out_path=f"{cfg.intermediate.genes}-{today}"
    )
'''