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
from includes.config import Config
import includes.model_functions as mf

def main(config):
    # df = pd.read_csv('model_data1.csv')
    df = pd.read_csv('tiny_one.csv')
    df.drop(df.columns[0], axis=1,inplace=True)
    X1_train, X1_test, y1_train, y1_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42)    
    #build tree_model
    all_tree_struc = mf.defined_all_trees(4)

    all_models = mf.build_single_models(mf.defined_all_models(4), X1_train)
    n=1
    for tree in all_tree_struc:
        mod_in_tree = [all_models.get(tuple(lst)) for lst in tree if tuple(lst) in all_models]
        tree_model = mod.tree_model('tree_mod1', mod_in_tree, tree)
        output = tree_model.predict(X1_test)
        print(f"Accuracy for tree {n} is: {accuracy_score(y1_test.to_list(), output['y_pred'].to_list())}")
        n+=1
    print(classification_report(y1_test.to_list(), output['y_pred'].to_list(), target_names=['1','2','3','4']))
    # dump(tree_model, config.model_path + '\\model.joblib')


if __name__ == '__main__':
    config = Config()
    main(config)