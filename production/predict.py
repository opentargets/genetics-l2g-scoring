#!/usr/bin/env python
'''
Predicts values for all loci in feature matrix.
'''
import argparse
import logging

import joblib
import pandas as pd
from pathlib import Path

# loading in pyspark is failing
# from pyspark import SparkContext
# from pyspark.mllib.tree import GradientBoostedTreesModel
# model = GradientBoostedTreesModel.load(sc, ROOT_DIR / MODEL_PATH)

ROOT_DIR = Path(__file__).parent.parent.absolute()


def main(
    all_loci: str,
    classifier: str,
    output_file: str,
) -> None:

    all_loci = pd.read_parquet(all_loci)
    classifier = joblib.load(ROOT_DIR / classifier)

    # Make predictions
    all_loci['y_proba'] = classifier['model'].predict_proba(
        all_loci
        # Keep only needed features
        .loc[:, classifier['run_info']['features']]
        # Recode True/False in has_sumstats to 1/0
        .replace({True: 1, False: 0})
    )[:, 1]

    # Save predictions
    l2g_predictions = format_predictions(all_loci)
    write_parquet(l2g_predictions, output_file)
    logging.info(f'Saved predictions to {output_file}')
    return None


def format_predictions(data: pd.DataFrame) -> pd.DataFrame:
    cols_keep = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id', 'y_proba']

    return data.loc[:, cols_keep].query('study_id != "GCST007236"')


def write_parquet(data: pd.DataFrame, output_file: str) -> None:
    Path.mkdir(Path(output_file).parents[0], exist_ok=True)
    data.to_parquet(output_file, compression='gzip')
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--all_loci',
        type=str,
        required=True,
        help='Feature matrix containing all loci. It is the output of `2_process_training_data`.',
    )
    parser.add_argument(
        '--classifier',
        type=str,
        required=True,
        help='Compressed XGBoost classifier. It is the output of `3_train_model`.',
    )
    parser.add_argument(
        '--output_file', type=str, required=True, help='Filename of the parquet file containing the L2G predictions.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    main(all_loci=args.all_loci, classifier=args.classifier, output_file=args.output_file)
