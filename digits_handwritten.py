import pandas as pd
# import sklearn as sk
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
import time
from itertools import combinations
import xgboost as xgb
from sklearn import svm
import warnings
from sklearn.exceptions import DataConversionWarning


def main(config):
    start_time = time.perf_counter()
    digits = datasets.load_digits()
    X = pd.DataFrame(digits.data)
    y = digits.target
    X['Y'] = y

    ## Read new data
    X = pd.read_csv('C:\\Users\\maxdi\\OneDrive\\Documents\\uni_honours_docs\\new_100k_10_cat.csv')
    X.rename({'label': 'Y'}, axis=1, inplace=True)
    y = X['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    # X_test = pd.read_csv('C:\\Users\\maxdi\\OneDrive\\Documents\\uni_honours_docs\\digits_test.csv')

    ## Best struct
    # found_best_mod_struc = [[(1, 3, 8, 9), (4,)], [(2,), (0, 1, 3, 4, 5, 6, 7, 8, 9)], [(3, 8), (1, 9)], [(1,), (9,)], [(1, 3, 4, 7, 8, 9), (5, 6)], [(1, 3, 4, 8, 9), (7,)], [(0,), (1, 3, 4, 5, 6, 7, 8, 9)], [(5,), (6,)], [(3,), (8,)]]
    # found_best_mod_struc = [[(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)], [(0, 1, 2), (3, 4)], [(5, 6, 7, 8), (9,)], [(0, 2), (1,)], [(3,), (4,)], [(5, 6), (7, 8)], [(0,), (2,)], [(5,), (6,)], [(7,), (8,)]]    
    # all_models_def = mf.build_single_models(found_best_mod_struc, X_train, train_type='xgboost')
    # # all_models_def = mf.build_single_models(found_best_mod_struc, X_train, train_type='Logistic')
    # all_models = list(all_models_def.values())
    # tree_model = mod.tree_model('tree_mod1', all_models, found_best_mod_struc)
    # print(tree_model.tree_struct)
    # output = tree_model.predict(X_test)
    # accuracy = accuracy_score(y_test.tolist(), output['y_pred'].to_list())
    # print(accuracy)
    # print(f'Took {time.perf_counter() - start_time}')
    # return

    ## Multiclass models
    # model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
    # model = xgb.XGBClassifier(objective='multi:softmax')
    # model = KNeighborsClassifier(n_neighbors=2)
    # model = svm.LinearSVC(dual="auto")

    # model.fit(X_train, y_train)

    # # Make predictions on the test set
    # y_pred = model.predict(X_test)

    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy: {accuracy:.4f}')
    # print(classification_report(y_test, y_pred, target_names=['0','1','2','3','4','5','6','7','8','9']))
    # print(f'Took {time.perf_counter() - start_time}')

    # return

    ## Stepwise Models
    warnings.filterwarnings(action='ignore')

    categories = tuple((0,1,2,3,4,5,6,7,8,9))
    # With model now decided, we can score and solve for accuracy
    stepwise_models = mf.stepwise_tree_layer_by_layer(categories, X_train, X_test, [])

    # With model now decided, we can score and solve for accuracy
    tree_names = [model.category_split for model in stepwise_models]
    normalized_tree = sorted(tree_names, key=len, reverse=True)
    tree_model = mod.tree_model('tree_mod1', stepwise_models, normalized_tree)
    print(tree_model.tree_struct)
    output = tree_model.predict(X_test)
    accuracy = accuracy_score(y_test.tolist(), output['y_pred'].to_list())
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_test.tolist(), output['y_pred'].to_list(), target_names=['0','1','2','3','4','5','6','7','8','9']))
    print(f'Took {time.perf_counter() - start_time}')

if __name__ == '__main__':
    config = Config()
    main(config)