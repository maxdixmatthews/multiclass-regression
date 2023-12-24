from joblib import dump, load
import pandas as pd
import includes.model as mod
from sklearn.model_selection import train_test_split

def save_model(path, model):
    """
    Saves a model to joblib file
    input:
        path: string to where file should be saved
        model: model to be saved
    output:
        returns none
    """
    dump(model, path)

def read_model(path: str):
    """
    Saves a model to joblib file
    input:
        path: string to where file should be saved
    output:
        returns none
    """
    return load(path)

def build_single_models(models_list: list, train_data) -> list:
    """
    Saves a model to joblib file
    input:
        models_list: list of 2 elements lists with models to be produced e.g [['123','4'],['13', '2']]
        train_data: data that will be used to train all models 
    output:
        returns list of models
    """
    trained_model_lists = dict()

    for i in models_list:
        new_mod = mod.single_model(i)
        new_mod.train(train_data)
        trained_model_lists[tuple(i)] = new_mod
    return trained_model_lists

def defined_all_models(n: int):
    """
    TODO remove this method and use something more rigourous. This function creates a list of all single models to be produced
    input:
        n: number of categories/classes
    output:
        list of binary comparisons
    """
    all_comparisons = list()
    # set(), sort, sort all strings 
    if n == 4:
        all_comparisons = [['1','2'], ['1','3'], ['1','4'], ['2','3'], ['3','4'], ['12','3'], ['12','4'], ['13','4'], 
                           ['13', '2'], ['23','4'], ['24','3'], ['123','4'], ['124','3'], ['134','2'], ['234','1']]
    return all_comparisons

def defined_all_trees(n: int):
    """
    TODO remove this method and use something more rigourous. This function creates a list of all trees or combined models for a given 
    number of categories. 
    input:
        n: number of categories/classes
    output:
        list of all trees
    """
    all_trees = list()
    if n == 4:
        all_trees = [[['123', '4'], ['13', '2'], ['1', '3']], 
                     [['12','34'], ['1', '2'], ['3', '4']],
                     [['124', '3'], ['12', '4'], ['1', '2']]]
    return all_trees
