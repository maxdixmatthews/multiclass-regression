import pandas as pd
import concurrent.futures
import numpy as np
import math
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, StratifiedKFold
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
import itertools


def main(filename, model_types):

    print(filename)
    if len(filename) <= 1:
        raise Exception(f"Improper filename of: {filename}")

    score_type = 'accuracy'
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    dataset = filename

    dataset_location = "data/" + dataset
    all_fold_score = {}
    all_fold_time = {}
    config = Config("all_trees_" + dataset + "_" + "_".join(model_types))
    df = pd.read_csv(dataset_location)
    df.drop(df.columns[0], axis=1, inplace=True)

    transform_label = mf.all_trees_map_categorical_target(config, df)
    categories = tuple(df['Y'].unique())

    trees_defined = mf.defined_all_trees(len(categories))
    
    unique_elements = set()
    for sublist in trees_defined:
        lil_one = list()
        for i in sublist:
            lil_one.append(tuple(i))
        unique_elements.add(tuple(lil_one))

    # Convert set back to list if needed
    unique_list = list(unique_elements)

    for fold, (train_index, test_index) in enumerate(cv.split(df, df['Y'])):
        start = time.perf_counter()
        X_full_train, X_full_test = df.iloc[train_index], df.iloc[test_index]
        y_full_train, y_full_test = df['Y'].iloc[train_index], df['Y'].iloc[test_index]
        print(f"Fold {fold+1}")
        X_train, X_test, y_train, y_test = train_test_split(X_full_train, X_full_train['Y'], stratify=X_full_train['Y'], test_size=0.3, random_state=42)
        best_accuracy = 0
        best_model = ()
        best_types = []

        elements = tuple(range(0, len(categories)))
        unique_list = list(mf.all_nested_dichotomies(elements))

        for tree in unique_list:
            all_possible_model_types = itertools.product(model_types, repeat=len(unique_list[0]))
            for tree_types in all_possible_model_types:
                model_strucs = tree
                
                try:
                    trained_model = mf.build_tree(config, X_test, X_train, y_test, score_type, list(tree_types), model_strucs, categories, transform_label=transform_label)[0]
                except Exception as e:
                    print("RED ALERT! RED ALERT!")
                    print(f"Failed at tree {tree}")
                    raise e

                if trained_model.score > best_accuracy:
                    best_accuracy = trained_model.score
                    best_model = tree
                    best_types = list(tree_types)
                    config.log.info(f'current best accuracy is {trained_model.score}')

        config.log.info(f'Best model is {best_model}')

        fold_time = round(time.perf_counter()-start,3)
        config.log.info(f"Fold {fold+1} took: {fold_time} to do find best tree.")
        all_fold_time[f"Fold {fold+1}"] = fold_time

        model_strucs = best_model
        config.log.info(model_strucs)
        config.log.info(best_types)
        best_trained_model = mf.build_best_tree(config, X_full_test, X_full_train, y_full_test, score_type, best_types, best_model, categories, transform_label=transform_label)[0]
        mf.graph_model(config, best_trained_model, f"all_trees_fold_{fold+1}_" + filename, transform_label=transform_label, model_types=model_types)
        config.log.info(f'Fold {fold+1}: {best_trained_model.score}')
        all_fold_score[f"Fold {fold+1}"] = best_trained_model.score

    average = sum(all_fold_score.values()) / len(all_fold_score)
    config.log.info(f"Average score: {average}")
    config.log.info(f"All the folds scores {all_fold_score}")
    config.log.info(f"All the folds time in seconds {all_fold_time}")
    for handler in config.log.handlers:
        handler.close()
        config.log.removeHandler(handler)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', required=True, type=str, help='The name of the file to process')
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['randomForest', 'LogisticRegression', 'xgboost'], help='An optional list models to be tested out of randomForest, LogisticRegression, xgboost, svm.')
    args = parser.parse_args()  
    main(args.filename, args.model_types)