#!/usr/bin/env python
'''
Predicts values for all loci in feature matrix.
'''
import logging
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils import *

# loading in pyspark is failing
# from pyspark import SparkContext
# from pyspark.mllib.tree import GradientBoostedTreesModel
# model = GradientBoostedTreesModel.load(sc, ROOT_DIR / MODEL_PATH)


@hydra.main(config_path=get_cwd(), config_name="config")
def main(cfg: DictConfig) -> None:

    # Parse args
    feature_matrix: str = cfg.l2g.feature_matrix
    classifier: str = cfg.l2g.model
    output_file: str = f"{cfg.l2g.output_bucket}/predictions-{datetime.now().strftime('%Y-%m-%d')}.parquet"

    # Load data and model
    feature_matrix = pd.read_parquet(feature_matrix)

    bucket_name, model_name = parse_gs_url(classifier)
    classifier = load_model(bucket_name, model_name)
    logging.info(f'{model_name} has been successfully loaded.')

    # Make predictions
    feature_matrix['y_proba'] = classifier['model'].predict_proba(
        feature_matrix
        # Keep only needed features
        .loc[:, classifier['run_info']['features']]
        # Recode True/False in has_sumstats to 1/0
        .replace({True: 1, False: 0})
    )[:, 1]

    # Save predictions
    l2g_predictions = format_predictions(feature_matrix)
    write_parquet(l2g_predictions, output_file)
    logging.info(f'Saved predictions to {output_file}')
    return None


def format_predictions(data: pd.DataFrame) -> pd.DataFrame:
    cols_keep = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id', 'y_proba']

    return data.loc[:, cols_keep].query('study_id != "GCST007236"')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    main()
