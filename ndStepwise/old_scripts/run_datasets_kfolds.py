import pandas as pd
import concurrent.futures
import numpy as np
import math
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, ConfusionMatrixDisplay, auc, roc_auc_score, f1_score
from sklearn import datasets
from statistics import mean
import includes.model as mod
import pandas as pd
from joblib import dump, load
from includes.config import Config;
import includes.model_functions as mf
import time
from itertools import combinations
import random
from graphviz import Digraph
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from datetime import datetime 
import os
import argparse
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

def main(filename, model_types):
    # config.log.info('Max Rocks')
    # config.log.error('This is an extra long message about how there was an error because Max wants to see if there is a weird format when messages get extra long.')
    # config.log.debug('THIS SHOULDNT LOG')
    # return
    # files = [
        # 'letter_recognition.csv',
        # 'car_evaluation.csv',
        # 'mfeat-factors.csv',
        # 'mfeat-fouriers.csv',
        # 'mfeat-karhunen.csv',
        # 'mfeat-morphological.csv',
        # 'mfeat-pixel.csv',
        # 'mfeat-zernlike.csv',
        # 'optdigits.csv',
        # 'pageblocks.csv',
        # 'handwritten_digits.csv',
        # 'satimage.csv',
        # 'image_segment.csv',
        # 'beans_data.csv',
# ]
    print(filename)
    if len(filename) <= 1:
        raise Exception(f"Improper filename of: {filename}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    dataset = filename
    # X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42)
    dataset_location = "data/" + dataset
    all_fold_score = []
    all_fold_time = []

    config = Config("inner_kfold_" + dataset + "_" + "_".join(model_types))
    df = pd.read_csv(dataset_location)
    df.drop(df.columns[0], axis=1, inplace=True)
    transform_label = mf.map_categorical_target(config, df)
    categories = tuple(df['Y'].unique())

    for fold, (train_index, test_index) in enumerate(cv.split(df, df['Y'])):
        start = time.perf_counter()
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = df['Y'].iloc[train_index], df['Y'].iloc[test_index]
        print(f"Fold {fold+1}")
    
        config.log.info(f'Beginning of {dataset}.')
        score_type = 'accuracy'

        config.log.info('Beginning of stepwise tree finder.')
        best_tree = mf.kfold_stepwise_tree_finder(config, categories, X_train, [], {}, model_types=model_types, score_type=score_type)
        config.log.info('Finished stepwise tree finder.')
        train_time = round(time.perf_counter()-start,3)
        config.log.info(f"Took: {train_time} to do find best tree.")
        model_strucs = list(best_tree.keys())
        tree_types = list(best_tree.values())
        config.log.info(model_strucs)
        config.log.info(tree_types)
        # kfold_build_best_tree()
        best_trained_model = mf.kfold_build_best_tree(config, X_test, X_train, y_test, score_type, tree_types, model_strucs, categories, transform_label=transform_label)[0]
        mf.graph_model(config, best_trained_model, f"inner_kfolds_fold{fold+1}_" + filename, transform_label=transform_label, model_types=model_types)
        config.log.info(f'Fold {fold+1}: {best_trained_model.score}')
        all_fold_score.append(best_trained_model.score)
        all_fold_time.append(train_time)

    # average = sum(all_fold_score) / len(all_fold_score)
    config.log.info(f"All scores: {all_fold_score}")
    config.log.info(f"All times: {all_fold_time}")
    for handler in config.log.handlers:
        handler.close()
        config.log.removeHandler(handler)
    logging.getLogger().handlers.clear()
    config.log = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('-f', '--filename', required=True, type=str, help='The name of the file to process')
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['randomForest', 'LogisticRegression', 'xgboost'], help='An optional list models to be tested out of randomForest, LogisticRegression, xgboost, svm.')
    args = parser.parse_args()
    main(args.model_types)