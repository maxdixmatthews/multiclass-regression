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


def main(filename, model_types):
    # config.log.info('Max Rocks')
    # config.log.error('This is an extra long message about how there was an error because Max wants to see if there is a weird format when messages get extra long.')
    # config.log.debug('THIS SHOULDNT LOG')
    # return
    print(filename)
    if len(filename) <= 1:
        raise Exception(f"Improper filename of: {filename}")
    dataset = filename
    config = Config(dataset)
    config.log.info(f'Beginning of {dataset}.')
    df = pd.read_csv(dataset)
    df.drop(df.columns[0], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42)
    score_type = 'accuracy'
    # categories = tuple((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))  
    categories = tuple(df['Y'].unique())

    config.log.info('Beginning of stepwise tree finder.')
    best_tree = mf.stepwise_tree_finder(config, categories, X_train, X_test, {}, model_types=model_types, score_type=score_type)
    config.log.info('Finished stepwise tree finder.')
    model_strucs = list(best_tree.keys())
    tree_types = list(best_tree.values())
    best_trained_model = mf.build_best_tree(config, X_test, X_train, y_test, score_type, tree_types, model_strucs, categories)
    mf.graph_model(config, best_trained_model, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', required=True, type=str, help='The name of the file to process')
    parser.add_argument('-m', '--model_types', type=str, nargs='*', default=['randomForest', 'LogisticRegression', 'xgboost'], help='An optional list models to be tested out of randomForest, LogisticRegression, xgboost, svm.')
    args = parser.parse_args()  
    main(args.filename, args.model_types)