import pandas as pd
# import sklearn as sk
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn import datasets
from statistics import mean
import includes.model as mod
import pandas as pd
from joblib import dump, load
from includes.config import Config;
import includes.model_functions as mf
import time
from itertools import combinations

def main(config):
    df = pd.read_csv('100k_6_cat.csv')
    df.drop(df.columns[0], axis=1,inplace=True)
    X1_train, X1_test, y1_train, y1_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42) 
    tree_len = 6
    all_tree_struc = mf.defined_all_trees(tree_len)
    all_model_struc = mf.single_models_from_trees(all_tree_struc)

    layer_1 = [tree for tree in all_model_struc if sum(len(t) for t in tree) == 6]
    all_layer1_models = mf.build_single_models(all_model_struc, X1_train)
    mf.test_single_models(all_layer1_models, X1_test, y1_test)
    # run all layer_1, then pick the best one, then find all trees under this layer 1
    print(layer_1)
    for tree in layer_1:

    #Now I have to go
    layer_2 = [tree for tree in all_model_struc if sum(len(t) for t in tree) == 6]

    #generate all first layer ones 

if __name__ == '__main__':
    config = Config()
    main(config)