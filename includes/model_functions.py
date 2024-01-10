from joblib import dump, load
import pandas as pd
import includes.model as mod
from itertools import combinations
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
    categories = tuple(range(1, n + 1))
    all_trees_normalized = generate_normalized_branches(categories)

    # Convert frozensets back to lists for readability
    all_trees_normalized_list = [sorted(list(map(list, tree))) for tree in all_trees_normalized]
    all_trees_normalized_list = [[sorted(branch, key=len, reverse=True) for branch in tree] for tree in all_trees_normalized_list]
    return all_trees_normalized_list

def stringify(node):
    """ Convert a tuple of numbers into a concatenated string. """
    return node
    return ''.join(map(str, node))

def generate_normalized_branches(categories):
    """
    Recursively generate all branches for the given categories with normalized order.
    This function ensures that each branch is represented in a standardized way to eliminate duplicates.
    """
    if len(categories) <= 1:
        return [set()]  # No branches can be formed from a single category

    branches_set = set()
    for left in generate_subsets(categories):
        right = tuple(set(categories) - set(left))

        # Generate branches for left and right subsets
        left_branches = generate_normalized_branches(left)
        right_branches = generate_normalized_branches(right)

        for l_branch_set in left_branches:
            for r_branch_set in right_branches:
                # Combine current split with left and right branches
                new_branch = tuple(sorted([stringify(left), stringify(right)]))
                combined_branches = {new_branch}.union(l_branch_set, r_branch_set)
                branches_set.add(frozenset(combined_branches))  # Using frozenset to allow set of sets
    return branches_set

def generate_subsets(s):
    """ Generate all non-empty subsets of a set s. """
    subsets = []
    for r in range(1, len(s)):
        subsets.extend(combinations(s, r))
    return subsets

def single_models_from_trees(trees_total):
    """Get list of all models to be generated from the trees"""
    total_models = [sorted(branch, key=lambda x: len(x), reverse=True)  for tree in trees_total for branch in tree]
    return_model = [list(t) for t in set(tuple(e) for e in total_models)]
    return return_model
