#!/usr/bin/env python
'''
Predicts values for all loci in feature matrix.
'''
import argparse
import logging

import gcsfs
import joblib
import pandas as pd
from pathlib import Path

# loading in pyspark is failing
# from pyspark import SparkContext
# from pyspark.mllib.tree import GradientBoostedTreesModel
# model = GradientBoostedTreesModel.load(sc, ROOT_DIR / MODEL_PATH)

#ROOT_DIR = Path(__file__).parent.parent.absolute()


def main(
    feature_matrix: str,
    classifier: str,
    output_file: str,
) -> None:

    # Load data and model
    feature_matrix = pd.read_parquet(feature_matrix).head()

    bucket_name, model_name = parse_gs_url(classifier)
    classifier = load_model(bucket_name, model_name)
    print('Loaded model')

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

def load_model(bucket_name, file_name):
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'{bucket_name}/{file_name}') as f:
        return joblib.load(f)

def parse_gs_url(gs_url):
    gs_url_split = gs_url.replace("gs://", "").split("/")
    return '/'.join(gs_url_split[:-1]), gs_url_split[-1]

def format_predictions(data: pd.DataFrame) -> pd.DataFrame:
    cols_keep = ['study_id', 'chrom', 'pos', 'ref', 'alt', 'gene_id', 'y_proba']

    return data.loc[:, cols_keep].query('study_id != "GCST007236"')


def write_parquet(data: pd.DataFrame, output_file: str) -> None:
    #Path.mkdir(Path(output_file).parents[0], exist_ok=True)
    data.to_parquet(output_file, compression='gzip')
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--feature_matrix',
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

    main(feature_matrix=args.feature_matrix, classifier=args.classifier, output_file=args.output_file)
