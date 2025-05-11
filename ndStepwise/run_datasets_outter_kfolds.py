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
import sqlalchemy as sa

def run_nd_stepwise(filename, model_types, kfold_seed):
    # config.log.info('Max Rocks')
    # config.log.error('This is an extra long message about how there was an error because Max wants to see if there is a weird format when messages get extra long.')
    # config.log.debug('THIS SHOULDNT LOG')
    # return
    # files = [
    #     'letter_recognition.csv',
    #     'car_evaluation.csv',
    #     'mfeat-factors.csv',
    #     'mfeat-fouriers.csv',
    #     'mfeat-karhunen.csv',
    #     'mfeat-morphological.csv',
    #     'mfeat-pixel.csv',
    #     'mfeat-zernlike.csv',
    #     'optdigits.csv',
    #     'pageblocks.csv',
    #     'handwritten_digits.csv',
    #     'satimage.csv',
    #     'image_segment.csv',
    #     'beans_data.csv',
    # ]
    # for filename in files:
    print(filename)
    print(model_types)
    if len(filename) <= 1:
        raise Exception(f"Improper filename of: {filename}")
    # start = time.perf_counter()
    kfold_seed = kfold_seed
    traversal_type = "ND-Stepwise"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=kfold_seed)
    dataset = filename
    # X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42)
    dataset_location = "data/" + dataset
    all_fold_score = {}
    all_fold_time = {}
    config = Config(dataset + "_" + "_".join(model_types))
    config.set_do_not_save_models()
    underscored_model_types = "_".join(model_types)
    df = pd.read_csv(dataset_location)
    df.drop(df.columns[0], axis=1, inplace=True)
    transform_label = mf.map_categorical_target(config, df)
    categories = tuple(df['Y'].unique())

    for fold, (train_index, test_index) in enumerate(cv.split(df, df['Y'])):
        start = time.perf_counter()
        X_train, X_test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = df['Y'].iloc[train_index], df['Y'].iloc[test_index]
        print(f"Fold {fold+1}")
        config.log.info(f'Beginning of fold {fold+1}  of {dataset}.')
        dataset_location = "data/" + dataset
        score_type = 'accuracy'
        config.log.info('Beginning of stepwise tree finder.')
        best_tree = mf.stepwise_tree_finder(config, categories, X_train, [], {}, model_types=model_types, score_type=score_type)
        config.log.info('Finished stepwise tree finder.')
        fold_time = round(time.perf_counter()-start,3)
        config.log.info(f"Fold {fold+1} took: {fold_time} to do find best tree.")
        all_fold_time[f"Fold {fold+1}"] = fold_time
        model_strucs = list(best_tree.keys())
        tree_types = list(best_tree.values())
        config.log.info(model_strucs)
        config.log.info(tree_types)

        best_trained_model, y_true, y_pred = mf.build_best_tree(config, X_test, X_train, y_test, score_type, tree_types, model_strucs, categories, transform_label=transform_label)
        # mf.graph_model(config, best_trained_model, f"kfold_{fold+1}" + filename, transform_label=transform_label, model_types=model_types)

        config.log.info(f'Fold {fold+1}: {best_trained_model.score}')
        all_fold_score[f"Fold {fold+1}"] = best_trained_model.score

        postgres_df = mf.calculate_metrics(config,y_true, y_pred)
        config.log.info(f'All metrics for fold-{fold+1}: {postgres_df}')
        run_timestamp =  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        id = traversal_type + underscored_model_types + dataset + "_" + "0" + str(fold+1) + "_" + str(kfold_seed) + run_timestamp
        postgres_df["id"] = id
        postgres_df["name"] = dataset + "_" + underscored_model_types
        postgres_df["models"] = underscored_model_types
        postgres_df["dataset"] = dataset[:-4]
        postgres_df["kfold"] = fold+1
        postgres_df["kfold_seed"] = kfold_seed
        postgres_df["nd_structure"] = str(model_strucs)
        postgres_df["model_structure"] = str(tree_types)
        postgres_df["run_time_seconds"] = fold_time
        postgres_df["inner_kfolds"] = 0
        postgres_df["run_timestamp"] = run_timestamp
        postgres_df["nd_traversal_type"] = traversal_type
        postgres_df["notes"] = ""
        engine = sa.create_engine(os.environ['ML_POSTGRESS_URL'] + "/max")
        mf.df_upsert_postgres(postgres_df, "complete_nd_model_registry", engine, schema=None, match_columns=["models","dataset","kfold","kfold_seed","inner_kfolds","nd_traversal_type"], insert_only=False)
        engine = None

    average = sum(all_fold_score.values()) / len(all_fold_score)
    config.log.info(f"Average score: {average}")
    config.log.info(f"All the folds scores {all_fold_score}")
    config.log.info(f"All the folds time in seconds {all_fold_time}")
    for handler in config.log.handlers:
        handler.close()
        config.log.removeHandler(handler)
    logging.getLogger().handlers.clear()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', required=True, type=str, help='The name of the file to process')
    parser.add_argument('-s', '--seed', default=69, type=int, help='The seed used for the outer kfolds')
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['randomForest', 'LogisticRegression', 'xgboost'], help='An optional list models to be tested out of randomForest, LogisticRegression, xgboost, svm.')
    args = parser.parse_args()
    run_nd_stepwise(args.filename, args.model_types, args.seed)