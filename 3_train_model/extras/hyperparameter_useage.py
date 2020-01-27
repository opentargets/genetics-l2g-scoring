#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Ed Mountjoy
#
'''
Takes trained models and predicts gold-standards
'''

import sys
import os
import json
import pandas as pd
from pprint import pprint
import joblib
from glob import glob

def main():

    pd.options.mode.chained_assignment = None

    # Args
    in_model_pattern = 'output/models_190903_1000/*.model.joblib.gz'
    out_path = 'results/models_190903_1000/hyperparamter_usage.tsv'

    # Create output folder
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Load all models and count hyperparamters
    param_counts = []
    model_files = glob(in_model_pattern)
    for c, model_file in enumerate(model_files):

        print('Processing model {} of {}...'.format(c, len(model_files)))

        # Load model
        model = joblib.load(model_file)

        # Add params
        for key, value in model['model'].best_params_.items():
            outrow = {
                'classifier_name': model['run_info']['classifier_name'],
                'feature_name': model['run_info']['feature_name'],
                'gold_standard_set': model['run_info']['gold_standard_set'],
                'fold_name': model['run_info']['fold_name'],
                'param_key': key,
                'param_value': value
            }
            param_counts.append(outrow)

        # if c == 10:
        #     break
        
    # Make df
    df = pd.DataFrame(param_counts)

    # Get counts
    counts = (
        df
        .groupby(['classifier_name', 'feature_name', 'gold_standard_set', 'param_key', 'param_value'])
        .size()
        .reset_index()
    )

    # Write
    counts.to_csv(out_path, sep='\t', index=None)
    
    return 0


    

if __name__ == '__main__':

    main()
