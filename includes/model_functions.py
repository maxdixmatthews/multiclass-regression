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
        models_list: list of 2 elements lists with models to be produced e.g [[(1,2,3),(4,)],[(1,3), (2,)]]
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
    list_of_counts = [0, 0, 10, 40, 50, 90, 100, 300, 500, 700, 1000, 1000, 1000, 1000, 1000, 1000]
    return list_of_counts[cat_num]

def single_step_wise(categories, X1_train, X1_test, model_type='LogisticRegression'):
    model_score = 0
    return_model_first = None
    model_list = dict()
    accepted_count = 0
    iterations = get_iterations_num(len(categories))
    model_type = 'svm'
    model_type = 'LogisticRegression'

    for i in categories:
        first_category = i
        left_category = tuple(x for x in categories if int(x) != first_category)
        right_category = tuple(x for x in categories if int(x) == first_category)
        first_model_struc = [[left_category, right_category]]

        first_model = build_single_models(first_model_struc, X1_train, train_type=model_type)
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
        prop_model = build_single_models(prop_next_model_struc, X1_train, train_type=model_type)
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

def stepwise_inclusion(left_list, right_list, X_train, X_test, train_type='LogisticRegression'): 
    # Returns best_model and best_score as tuple (best_model, best_score)
    model_list = []
    for i in right_list:
        all_but_one = tuple(x for x in right_list if x != i)
        model_def = [tuple(tuple(left_list) + (i,)), all_but_one]
        model_list.append(model_def)
    
    model = build_single_models(model_list, X_train, train_type=train_type)
    tested_mods = test_single_models(model,X_test)
    sorted_d_desc = sorted(tested_mods.items(), key=lambda item: item[1], reverse=True)
    best_mod = sorted_d_desc[0][0]
    best_score = sorted_d_desc[0][1]
    return best_mod, best_score
    
def stepwise_exclusion(left_list, right_list, X_train, X_test, train_type='LogisticRegression'):
    # Returns best_model and best_score as tuple (best_model, best_score)
    model_list = []
    for i in left_list:
        all_but_one = tuple(x for x in left_list if x != i)
        model_def = [tuple(tuple(right_list) + (i,)), all_but_one]
        model_list.append(model_def)
    
    model = build_single_models(model_list, X_train, train_type=train_type)
    tested_mods = test_single_models(model, X_test)
    sorted_d_desc = sorted(tested_mods.items(), key=lambda item: item[1], reverse=True)
    best_mod_ordered = (sorted_d_desc[0][0][1], sorted_d_desc[0][0][0])
    best_score = sorted_d_desc[0][1]
    return best_mod_ordered, best_score
    
def stepwise_single_layer(categories, X_train, X_test, model_type='LogisticRegression'):
    best_mod, best_score = stepwise_inclusion([], categories, X_train, X_test)
    while True:
        new_mod, new_score = stepwise_inclusion(best_mod[0], best_mod[1], X_train, X_test, model_type)
        if new_score > best_score:
            best_mod = new_mod
            best_score = new_score
        else:
            if len(new_mod[1]) > 1:
                second_inclusion_mod, second_inclusion_score = stepwise_inclusion(new_mod[0], new_mod[1], X_train, X_test, model_type)
                if second_inclusion_score > best_score:
                    best_mod = second_inclusion_mod
                    best_score = second_inclusion_score
                else:
                    if len(second_inclusion_mod[1]) > 1:
                        third_inclusion_mod, third_inclusion_score = stepwise_inclusion(second_inclusion_mod[0], second_inclusion_mod[1], X_train, X_test, model_type)
                        if third_inclusion_score > best_score:
                            best_mod = third_inclusion_mod
                            best_score = third_inclusion_score
                        else:
                            break
                    else:
                        break
            else:
                break
        new_backward_mod, new_backward_score = stepwise_exclusion(best_mod[0], best_mod[1], X_train, X_test, model_type)
        if new_backward_score > best_score:
            best_mod = new_backward_mod
            best_score = new_backward_score      
        else:
            continue
    return best_mod

def stepwise_layer_finder(categories, X_train, X_test, model_type='LogisticRegression'):
    best_mod, best_score = new_mod, new_score = stepwise_inclusion([], categories, X_train, X_test)
    failed_model_counter = 0
    run_inclusion = True
 
    while True:
        if failed_model_counter > 3:
            break
        elif failed_model_counter == 0:
            best_mod = run_mod = new_mod
            best_score = run_score = new_score
        else:
            run_mod = new_mod
            run_score = new_score       
        if run_inclusion:
            new_mod, new_score = stepwise_inclusion(run_mod[0], run_mod[1], X_train, X_test, model_type)
            if new_score > best_score:
                best_mod = new_mod
                best_score = new_score
                run_inclusion = False
                failed_model_counter = 0
            else:
                run_inclusion = True
                failed_model_counter += 1
        else:
            new_mod, new_score = stepwise_exclusion(run_mod[0], run_mod[1], X_train, X_test, model_type)
            if new_score > best_score:
                best_mod = new_mod
                best_score = new_score      
                run_inclusion = True
                failed_model_counter = 0
                if len(new_mod[0]) > 1:
                    run_inclusion = False
            else:
                run_inclusion = True
                failed_model_counter += 1
    return best_mod

def stepwise_tree_finder(categories, X1_train, X1_test, total_tree):
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
            two_cat_mod = ((categories[0],), (categories[1],))
            return list(set(total_tree + [two_cat_mod]))
        else:
            return total_tree
    
    highest_model = stepwise_layer_finder(categories, X1_train, X1_test)
    total_tree.append(highest_model)

    total1 = stepwise_tree(tuple(highest_model[0]), X1_train, X1_test, total_tree)
    total2 = stepwise_tree(tuple(highest_model[1]), X1_train, X1_test, total1+total_tree)

    return list(set(total_tree + total2))

def stepwise_tree(categories, X1_train, X1_test, total_tree):
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
            two_cat_mod = ((categories[0],), (categories[1],))
            return list(set(total_tree + [two_cat_mod]))
        else:
            return total_tree
    
    highest_model = stepwise_single_layer(categories, X1_train, X1_test)
    total_tree.append(highest_model)

    total1 = stepwise_tree(tuple(highest_model[0]), X1_train, X1_test, total_tree)
    total2 = stepwise_tree(tuple(highest_model[1]), X1_train, X1_test, total1+total_tree)

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
