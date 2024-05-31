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


def main():
    # config.log.info('Max Rocks')
    # config.log.error('This is an extra long message about how there was an error because Max wants to see if there is a weird format when messages get extra long.')
    # config.log.debug('THIS SHOULDNT LOG')
    # return
    list_of_data = ["new_100k_10_cat.csv", "new_100k_6_cat.csv"]
    for dataset in list_of_data:
        process_dataset(dataset)

def process_dataset(dataset):
    config = Config(dataset)
    config.log.info(f'Beginning of {dataset}.')
    df = pd.read_csv(dataset)
    df.drop(df.columns[0], axis=1, inplace=True)
    # df = df.head(1000)
    X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42)
    score_type = 'accuracy'
    # categories = tuple((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))  
    categories = tuple(df['Y'].unique())

    model_types = ['randomForest', 'LogisticRegression', 'xgboost']
    config.log.info('Beginning of stepwise tree finder.')
    best_tree = mf.stepwise_tree_finder(config, categories, X_train, X_test, {}, model_types=model_types, score_type=score_type)
    config.log.info('Finished stepwise tree finder.')
    model_strucs = list(best_tree.keys())
    tree_types = list(best_tree.values())
    best_trained_model = mf.build_best_tree(config, X_test, X_train, y_test, score_type, tree_types, model_strucs, categories)
    mf.graph_model(config, best_trained_model)

if __name__ == '__main__':
    main()