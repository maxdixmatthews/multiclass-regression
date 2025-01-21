from sklearn.model_selection import train_test_split
import pandas as pd
from ndStepwise.includes.config import Config
from ndStepwise.includes import model_functions as mf
import time
import argparse
import ast
from tensorflow.keras.datasets import mnist

def main(filename, model_types, tree_structure):
    # config.log.info('Max Rocks')
    # config.log.error('This is an extra long message about how there was an error because Max wants to see if there is a weird format when messages get extra long.')
    # config.log.debug('THIS SHOULDNT LOG')
    # return
    print(filename)
    if len(filename) <= 1:
        raise Exception(f"Improper filename of: {filename}")
    start = time.perf_counter()
   
    dataset = filename
    config = Config(dataset)
    config.log.info(f'Beginning of {dataset}.')

    if dataset.lower() == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Flatten the 28x28 images into a single 784-length vector per image
        X_train_flattened = X_train.reshape(X_train.shape[0], -1)
        X_test_flattened = X_test.reshape(X_test.shape[0], -1)

        # Convert to a Pandas DataFrame
        X_train_df = pd.DataFrame(X_train_flattened)
        X_test_df = pd.DataFrame(X_test_flattened)

        # Optionally, add column names (e.g., "pixel_0", "pixel_1", ..., "pixel_783")
        X_train_df.columns = [f"pixel_{i}" for i in range(X_train_flattened.shape[1])]
        X_test_df.columns = [f"pixel_{i}" for i in range(X_test_flattened.shape[1])]

        # Add the labels as a separate column if desired
        X_train_df['Y'] = y_train
        X_test_df['Y'] = y_test
        X_train = X_train_df
        X_test = X_test_df
        y_test = y_test

        categories = tuple(X_train_df['Y'].unique())
        transform_label = None
    else:
        dataset_location = "data/" + dataset
        df = pd.read_csv(dataset_location)
        df.drop(df.columns[0], axis=1, inplace=True)
        transform_label = mf.map_categorical_target(config, df)
        X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], stratify=df['Y'], test_size=0.2, random_state=42)
        categories = tuple(df['Y'].unique())
    score_type = 'accuracy' 
    model_strucs = list(tree_structure) 
    tree_types = list(model_types)

    config.log.info('Testing individual tree.')
    config.log.info(model_strucs)
    config.log.info(tree_types)

    best_trained_model = mf.build_best_tree(config, X_test, X_train, y_test, score_type, tree_types, model_strucs, categories, transform_label=transform_label)[0]
    mf.graph_model(config, best_trained_model, filename, transform_label=transform_label, model_types=model_types)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-f', '--filename', required=True, type=str, help='The name of the file to process')
    parser.add_argument('-m', '--model_types', type=str, default=['randomForest', 'LogisticRegression', 'xgboost'], help='An optional list models to be tested out of randomForest, LogisticRegression, xgboost, svm.')
    parser.add_argument('-t', '--tree_structure', type=str, help='A list of the tree to be made.')
    args = parser.parse_args()  
    main(args.filename, ast.literal_eval(args.model_types), ast.literal_eval(args.tree_structure))

# "[((3,), (2, 0, 1)), ((2, 0), (1,)), ((2,), (0,))]"
# "['svm', 'svm', 'svm']""