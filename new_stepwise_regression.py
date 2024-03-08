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
import random

def main(config):
    df = pd.read_csv('new_100k_10_cat.csv') 
    categories = tuple((0,1,2,3,4,5,6,7,8,9))

    # df = pd.read_csv('100k_6_cat.csv')   
    # categories = tuple((1,2,3,4,5,6))
    df.drop(df.columns[0], axis=1,inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.2, random_state=42) 
    # stepwise_models = mf.stepwise_tree_layer_by_layer(categories, X1_train, X1_test, [])
    start_time = time.perf_counter()
    # best_tree = mf.stepwise_tree(categories, X_train, X_test, [])
    best_tree = [((3,), (0, 1, 2, 4, 6, 7, 9)), ((7,), (6,)), ((9,), (7, 6)), ((8,), (0, 1, 2, 3, 4, 5, 6, 7, 9)), ((4,), (0, 1, 2)), ((1,), (0, 2)), ((5,), (0, 1, 2, 3, 4, 6, 7, 9)), ((0,), (2,)), ((7, 6, 9), (0, 1, 2, 4))]
    best_tree = [[(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)], [(0, 1, 2), (3, 4)], [(5, 6, 7, 8), (9,)], [(0, 2), (1,)], [(3,), (4,)], [(5, 6), (7, 8)], [(0,), (2,)], [(5,), (6,)], [(7,), (8,)]]    

    normalized_tree = sorted(best_tree, key=len, reverse=True)
    built_mods = mf.build_single_models(normalized_tree, X_train)
    built_mods = list(built_mods.values())
    tree_model = mod.tree_model('tree_mod1', built_mods, normalized_tree)
    print(tree_model.tree_struct)
    output = tree_model.predict(X_test)
    accuracy = accuracy_score(y_test.tolist(), output['y_pred'].to_list())
    print(classification_report(y_test.tolist(), output['y_pred'].to_list(), target_names=['0','1','2','3','4','5','6','7','8','9']))
    print(accuracy)
    print(f'Took {time.perf_counter() - start_time}')
    return
    categories = tuple((5,6))
    best_mod, best_score = stepwise_inclusion([], categories, X_train, X_test)
    while True:
        new_mod, new_score = stepwise_inclusion(best_mod[0], best_mod[1], X_train, X_test)
        if new_score > best_score:
            best_mod = new_mod
            best_score = new_score
        else:
            if len(new_mod[1]) > 1:
                second_inclusion_mod, second_inclusion_score = stepwise_inclusion(new_mod[0], new_mod[1], X_train, X_test)
                if second_inclusion_score > best_score:
                    best_mod = second_inclusion_mod
                    best_score = second_inclusion_score
                else:
                    break
            else:
                break
            
        new_backward_mod, new_backward_score = stepwise_exclusion(best_mod[0], best_mod[1], X_train, X_test)
        if new_backward_score > best_score:
            best_mod = new_backward_mod
            best_score = new_backward_score      
        else:
            break
    print(best_mod)
    print(best_score)
    return
        
    while True:
        best_mod, best_score = stepwise_inclusion([], categories, X_train, X_test)
        left_list = best_mod[0]
        right_list = best_mod[1]
        current_best_score = best_score
        current_best_mod = best_mod

        new_mod, new_score = stepwise_inclusion(left_list, right_list, X_train, X_test)
        print(f'best score is {current_best_score} with model {current_best_mod}')
        if current_best_score > new_score:
            current_best_score = new_score
            current_best_mod = new_mod
            # break 
        else:
            current_best_score = new_score
            current_best_mod = new_mod

        left_list = new_mod[0]
        right_list = new_mod[1]

        new_mod, new_score = stepwise_inclusion(left_list, right_list, X_train, X_test)
        print(f'best score is {current_best_score} with model {current_best_mod}')
        if current_best_score > new_score:
            current_best_score = new_score
            current_best_mod = new_mod
            # break 
        else:
            current_best_score = new_score
            current_best_mod = new_mod

        left_list = new_mod[0]
        right_list = new_mod[1]

        new_mod, new_score = stepwise_inclusion(left_list, right_list, X_train, X_test)
        print(f'best score is {current_best_score} with model {current_best_mod}')
        if current_best_score > new_score:
            current_best_score = new_score
            current_best_mod = new_mod
            # break 
        else:
            current_best_score = new_score
            current_best_mod = new_mod

        break     

    return 
    while True:
        model_list = []
        for i in categories:
            all_but_one = tuple(x for x in categories if x != i)
            model_def = [(i,),all_but_one]
            model_list.append(model_def)
        
        model = mf.build_single_models(model_list, X_train, train_type='LogisticRegression')
        tested_mods = mf.test_single_models(model,X_test)
        sorted_d_desc = sorted(tested_mods.items(), key=lambda item: item[1], reverse=True)
        best_mod = sorted_d_desc[0][0]
        best_score = sorted_d_desc[0][1]
        print(sorted_d_desc)
        print(best_score)

        model_list = []
        left_list = best_mod[0]
        right_list = best_mod[1]
        for i in right_list:
            all_but_one = tuple(x for x in right_list if x != i)
            model_def = [tuple(left_list + (i,)),all_but_one] 
            model_list.append(model_def)           
        
        model = mf.build_single_models(model_list, X_train, train_type='LogisticRegression')
        tested_mods = mf.test_single_models(model,X_test)
        sorted_d_desc = sorted(tested_mods.items(), key=lambda item: item[1], reverse=True)
        best_mod = sorted_d_desc[0][0]
        best_score = sorted_d_desc[0][1]

        model_list = []
        left_list = best_mod[0]
        right_list = best_mod[1]
        for i in right_list:
            all_but_one = tuple(x for x in right_list if x != i)
            model_def = [tuple(left_list + (i,)),all_but_one] 
            model_list.append(model_def)           
        
        model = mf.build_single_models(model_list, X_train, train_type='LogisticRegression')
        tested_mods = mf.test_single_models(model,X_test)
        sorted_d_desc = sorted(tested_mods.items(), key=lambda item: item[1], reverse=True)
        best_mod = sorted_d_desc[0][0]
        best_score = sorted_d_desc[0][1]

        model_list = []
        left_list = best_mod[0]
        right_list = best_mod[1]
        print(best_mod)
        print(best_score)
        # do a 1 vs all calc for all of them 
        # add each to a big list 
        # Select the best one 
        break

    # With model now decided, we can score and solve for accuracy
    # tree_names = [model.category_split for model in stepwise_models]
    # normalized_tree = sorted(tree_names, key=len, reverse=True)
    # tree_model = mod.tree_model('tree_mod1', stepwise_models, normalized_tree)
    # print(tree_model.tree_struct)
    # output = tree_model.predict(X1_test)
    # accuracy = accuracy_score(y1_test.to_list(), output['y_pred'].to_list())
    # print(accuracy)

if __name__ == '__main__':
    config = Config()
    main(config)