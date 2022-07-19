#!/usr/bin/env python
'''
Creates features from the colocalization of mQTLs.
'''
import logging
from datetime import datetime

from hydra import main
from omegaconf import DictConfig
import pyspark.sql.functions as F

from src.utils import *


@main(config_path=f'{get_cwd()}/config', config_name="config")
def main(cfg: DictConfig) -> None:

    process_sumstat_coloc(
        coloc_path=cfg.feature_extraction.processed_inputs.coloc,
        credset_path=cfg.feature_extraction.processed_inputs.credset_qtl,
    )


def process_sumstat_coloc(coloc_path: str, credset_path: str):
    """
    Processing of colocalization analyses of GWAS summary statistics.
    """

    qtl_df = (
        spark.read.parquet(credset_path)
        .withColumnRenamed('study_id', 'right_study')
        .withColumnRenamed('bio_feature', 'right_bio_feature')
        .withColumnRenamed('phenotype_id', 'right_phenotype')
        .withColumnRenamed('lead_chrom', 'right_chrom')
        .withColumnRenamed('lead_pos', 'right_pos')
        .withColumnRenamed('lead_ref', 'right_ref')
        .withColumnRenamed('lead_alt', 'right_alt')
        .filter(F.col('is_sentinel'))
        .select(
            'right_study',
            'right_bio_feature',
            'right_phenotype',
            'right_chrom',
            'right_pos',
            'right_ref',
            'right_alt',
            'qtl_neglog_p',
        )
        .distinct()
    )

    coloc_eqtl_df = (
        spark.read.parquet(coloc_path)
        .filter(F.col('qtl_type') == 'eqtl')
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
        .persist()
    )

    pass


if __name__ == '__main__':

    # Initialise spark and logger
    logging.basicConfig(level=logging.INFO)
    spark = initialize_sparksession()

    main()
