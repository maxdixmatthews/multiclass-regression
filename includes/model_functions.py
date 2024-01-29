from joblib import dump, load
import pandas as pd
import includes.model as mod
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random 
import numpy as np


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

def build_single_models(models_list: list, train_data, score_type='accuracy', train_type='LogisticRegression') -> list:
    """
    Builds all single models
    input:
        models_list: list of 2 elements lists with models to be produced e.g [['123','4'],['13', '2']]
        train_data: data that will be used to train all models 
        score_type: The metric we are looking to maximuse
    output:
        returns list of models
    """
    trained_model_lists = dict()

    for i in models_list:
        new_mod = mod.single_model(i, score_type=score_type)
        new_mod.train(train_data, model_type = train_type)
        trained_model_lists[tuple(i)] = new_mod
    return trained_model_lists

def test_single_models(models: list, x_test_data):
    """
    Check all single models
    """
    tested_models = dict()
    for key in models:
        model = models[key]
        model.predict_individual(x_test_data)
        tested_models[key] = model.model_score()
    return tested_models

def find_best_mod_given_categories(categories, all_model_struc, X1_train, X1_test):
    """
    Finds the split of a model that has the highest accuracy/f1/metric
    input:
        categories: a tuple of the lists 
        all_model_struc: a list of all models
        X1_train: the x train data
        X1_test: the x test data
        total_tree: a list of the biggest models
    output:
        The top model
    """  
    layered_models = [model for model in all_model_struc if sorted(list(categories)) == sorted(list(set(model[0]+model[1])))]
    all_layer_models = build_single_models(layered_models, X1_train)

    scores_2_r = test_single_models(all_layer_models, X1_test)
    highest_model = all_layer_models[max(scores_2_r, key=scores_2_r.get)] 
    return highest_model

def build_tree_layer_by_layer(categories, all_model_struc, X1_train, X1_test, total_tree):
    """
    Build this tree in a recursive way
    input:
        categories: a tuple of the lists 
        all_model_struc: a list of all models
        X1_train: the x train data
        X1_test: the x test data
        total_tree: a list pf the biggest models
    output:
        list of binary comparisons
    """
    if len(categories) < 2:
        return total_tree
    
    highest_model = find_best_mod_given_categories(categories, all_model_struc, X1_train, X1_test)
    total_tree.append(highest_model)

    total1 = build_tree_layer_by_layer(highest_model.type_0_categories, all_model_struc, X1_train, X1_test, total_tree)
    total2 = build_tree_layer_by_layer(highest_model.type_1_categories, all_model_struc, X1_train, X1_test, total1+total_tree)

    return list(set(total_tree + total2))

def get_iterations_num(cat_num:int):
    """
    Get the number of iterations needed for a given number of categories 
    input:
        cat_num: int of number of categories
    output:
        returns int of how many iterations needed for stepwise
    """
    list_of_counts = [0,0,10,40,50,90,100,300,500,500,500]
    return list_of_counts[cat_num]

def single_step_wise(categories, X1_train, X1_test):
    model_score = 0
    return_model_first = None
    model_list = dict()
    accepted_count = 0
    iterations = get_iterations_num(len(categories))
    for i in categories:
        first_category = i
        left_category = tuple(x for x in categories if int(x) != first_category)
        right_category = tuple(x for x in categories if int(x) == first_category)
        first_model_struc = [[left_category, right_category]]

        first_model = build_single_models(first_model_struc, X1_train)
        prop_model_score = list(test_single_models(first_model, X1_test).values())[0]
        return_model = first_model
        if prop_model_score > model_score:
            return_model_first = return_model
            model_score = prop_model_score
            model_list[prop_model_score] = list(first_model.values())[0]
    

    for i in range(0,iterations):
        select_category_int = random.choice(tuple((0,1)))
        random_noise = np.random.normal(loc=0, scale=0.06)
        #TODO cleanup this if mess

        if (select_category_int == 0 and len(left_category) >= 2) or (len(right_category) == 1):
            # Select the left category
            selected_cat = left_category
            next_category = random.choice(selected_cat)
            prop_left_category = tuple(x for x in categories if x in left_category and int(x) != next_category)
            prop_right_category = tuple(x for x in categories if x in right_category or int(x) == next_category)
        elif (select_category_int == 1 and len(right_category) >= 2) or (len(left_category) == 1):
            # Select the right category
            selected_cat = right_category
            next_category = random.choice(selected_cat)
            prop_right_category = tuple(x for x in categories if x in right_category and int(x) != next_category)
            prop_left_category = tuple(x for x in categories if x in left_category or int(x) == next_category)

        # print(f'Prop left things {prop_left_category}')
        # print(f'Prop right things {prop_right_category}')
        prop_next_model_struc = [[prop_left_category, prop_right_category]]
        # print(prop_next_model_struc)
        prop_model = build_single_models(prop_next_model_struc, X1_train)
        prop_model_score = list(test_single_models(prop_model, X1_test).values())[0]

        if model_score < prop_model_score + random_noise:
            model_score = prop_model_score
            return_model = prop_model
            left_category = prop_left_category
            right_category = prop_right_category
            accepted_count += 1
            model_list[prop_model_score] = list(prop_model.values())[0]
    values = [x for x in model_list]
    return(model_list.get(max(values)))

def stepwise_tree_layer_by_layer(categories, X1_train, X1_test, total_tree):
    """
    Build this tree in a stepwise recursive way
    input:
        categories: a tuple of the lists 
        all_model_struc: a list of all models
        X1_train: the x train data
        X1_test: the x test data
        total_tree: a list pf the biggest models
    output:
        list of binary comparisons
    """
    if len(categories) <= 2:
        if len(categories) == 2:
            two_cat_mod = list(build_single_models([[(categories[0],), (categories[1],)]], X1_train).values())[0]
            return list(set(total_tree + [two_cat_mod]))
        else:
            return total_tree
    
    highest_model = single_step_wise(categories, X1_train, X1_test)
    total_tree.append(highest_model)

    total1 = stepwise_tree_layer_by_layer(highest_model.type_0_categories, X1_train, X1_test, total_tree)
    total2 = stepwise_tree_layer_by_layer(highest_model.type_1_categories, X1_train, X1_test, total1+total_tree)

    return list(set(total_tree + total2))


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
    categories = tuple(range(1, n+1))
    all_trees_normalized = generate_normalized_branches(categories)

    # Convert frozensets back to lists for readability
    all_trees_normalized_list = [sorted(list(map(list, tree))) for tree in all_trees_normalized]
    all_trees_normalized_list = [[sorted(branch, key=len, reverse=True) for branch in tree] for tree in all_trees_normalized_list]
    return all_trees_normalized_list

def stringify(node):
    """ Convert a tuple of numbers into a concatenated string. """
    return node

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
                new_branch = tuple(sorted([left, right]))
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
