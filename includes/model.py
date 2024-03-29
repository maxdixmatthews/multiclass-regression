from abc import ABC, abstractmethod, abstractproperty
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import svm

class model(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
    
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass

class single_model(model):
    # Name a model (x,)_(y,) to do x vs y regression pass list [(x,),(y,)]. 
    # E.g (1,2,3)_(4,5,6) to do a binary model of 123 vs 456 
    # 123 = 0 and 456 = 1
    def __init__(self, category_split: list, score_type='accuracy'):
        num_list_1 = [int(digit) for digit in category_split[0]]
        num_list_2 = [int(digit) for digit in category_split[1]]
        name = str(category_split[0]) + '_' + str(category_split[1])
        super().__init__(name)
        self.name = name
        self.all_cat_tested = list(set(num_list_1 + num_list_2))
        self.type_0_categories = category_split[0]
        self.type_1_categories = category_split[1]
        self.type_0 = num_list_1
        self.type_1 = num_list_2
        self.fitted_model = None
        self.predicted_df = None
        self.category_split = category_split
        self.score = None
        self.y_target = None
        if score_type == 'accuracy':
            self.score_type = 'accuracy'
        else:
            self.score_type = 'accuracy'
    def get_model_name(self):
        return self.name
    
    def train(self, df: pd.DataFrame, model_type = 'LogisticRegression', response_col = 'Y'):
        """
        Trains the model on a given set of data 
        input:
            df: pandas dataframe of the train data
            response_col: optional name of the column with response in it (defauls to Y)
        output:
            sds
        """
        # Setup
        train_df = df.copy()
        train_df[self.name] = train_df[response_col].apply(lambda x: 0 if x in self.type_0 else (1 if x in self.type_1 else 'ROW_NOT_IN_REG') )
        train_df = train_df.loc[train_df[self.name] != 'ROW_NOT_IN_REG']
        Y = train_df[self.name].astype(int)

        #Select model
        if model_type == 'LogisticRegression':
            model = LogisticRegression(solver='lbfgs', max_iter=2000)
        elif model_type.lower() == 'xgboost':
            model = xgb.XGBClassifier(objective="binary:logistic")
            # model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
            # model = KNeighborsClassifier(n_neighbors=5)
            # Will have to do hyperparameter tuning
            # Define search space
            # search_spaces = {   
            #     'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            #     'max_depth': Integer(2, 20),
            #     'reg_lambda': Real(1e-9, 100., 'log-uniform'),
            #     'reg_alpha': Real(1e-9, 100., 'log-uniform'),
            #     'gamma': Real(1e-9, 0.5, 'log-uniform'),  
            #     'n_estimators': Integer(10, 1000)
            # }
            # model = BayesSearchCV(
            #                     estimator = xgb_model,                                    
            #                     search_spaces = search_spaces,                      
            #                     scoring = 'roc_auc',                                  
            #                     cv = StratifiedKFold(n_splits=5, shuffle=True),                                   
            #                     n_iter = 20,                                      
            #                     n_points = 5,                                       
            #                     n_jobs = 1,                                                                                
            #                     verbose = 0,
            #                     random_state=42,
            #                     refit=True
            # )  
            np.int = int
            # _ = bayes_cv.fit(train_df.drop([response_col,self.name], axis=1), Y)
            # model = xgb.XGBClassifier(
            #     n_jobs = 5,
            #     objective = 'binary:logistic',
            #     eval_metric = 'auc', 
            #     booster = 'gbtree',
            #     enable_categorical = True, 
            #     early_stopping_rounds = 5,
            #     **bayes_cv.best_params_
            # )
        elif model_type.lower() == 'svm':
            model = svm.SVC()
        else:
            model = LogisticRegression(solver='sag', max_iter=2000)
            # XGBoost, Neural Network, Stukel model, anyother will work 
            # beat multinomial regression 
        model.fit(train_df.drop([response_col,self.name], axis=1), Y)
        self.fitted_model = model
        return model

    def predict(self, df_original: pd.DataFrame):
        """
        Tests the model on a given set of data. Must be called after a model has been trained
        input:
            df: pandas dataframe of the test data
            response_col: optional name of the column with response in it (defauls to Y)
        output:
            sds
        """
        df = df_original.copy()
        df['key'] = df.index
        response_col = 'Y'
        # train_df = df.copy()
        # train_df[self.name] = train_df[response_col].apply(lambda x: 0 if x in self.type_0 else (1 if x in self.type_1 else 'ROW_NOT_IN_REG') )
        # train_df = train_df.loc[train_df[self.name] != 'ROW_NOT_IN_REG']
        if self.fitted_model == None:
            raise Exception('Must train model on data before it can be tested.')
        y_pred = self.fitted_model.predict(df.drop(['key',response_col], axis=1))
        df['y_pred'] = y_pred
        self.predicted_df = df[['key','y_pred']]
        return self.predicted_df

    def predict_individual(self, df_original: pd.DataFrame):
        """
        Tests the model on a given set of data and scores the model independently of the others
        input:
            df: pandas dataframe of the test data
            response_col: optional name of the column with response in it (defauls to Y)
        output:
            sds
        """
        df = df_original.copy()
        df['key'] = df.index
        response_col = 'Y'
        train_df = df.copy()
        train_df[self.name] = train_df[response_col].apply(lambda x: 0 if x in self.type_0 else (1 if x in self.type_1 else 'ROW_NOT_IN_REG') )
        train_df = train_df.loc[train_df[self.name] != 'ROW_NOT_IN_REG']
        if self.fitted_model == None:
            raise Exception('Must train model on data before it can be tested.')
        y_pred = self.fitted_model.predict(train_df.drop(['key',response_col, self.name], axis=1))
        train_df['y_pred'] = y_pred
        self.predicted_df = train_df[['key','y_pred']]
        self.y_pred = y_pred
        self.y_target = train_df[self.name].tolist()
        
        return self.predicted_df
    
    def model_score(self):
        if self.y_pred is None or self.y_target is None:
            raise Exception ('Must run predict on a model before scoring it.')
        else:
            if self.score_type == 'accuracy':
                self.score = accuracy_score(self.y_target,self.y_pred.tolist())
            else:
                self.score = accuracy_score(self.y_target,self.y_pred.tolist())
            return self.score
    
    def get_prediction(self):
        return self.predicted_df

class tree_model(model):
    # name of model needs to be recognizable or some combination of the submodels
    def __init__(self, name, model_list, tree_struct):
        """
        
        input:
            name: name of this model
            model_list: list of all models in order that make up the regression
        output:
            sds
        """
        super().__init__(name)
        self.models = sorted(model_list, key=lambda x: len(x.name), reverse=True)
        #sort model list
        self.predicted_df = None
        self.tree_struct = tree_struct

    
    def train(self):
        """
        By design we feed tree model pretrained models
        """
        pass

    def predict(self, df:pd.DataFrame):
        predicted_dfs = dict()
        df_key = df.copy()
        df_key['key'] = df_key.index
        for model in self.models:
            df_pred = df.copy()
            y_pred = model.predict(df_pred)
            predicted_dfs.update({model:y_pred}) 

        df_key['total_pred'] = None

        # for index, row in df_key.iterrows():
        #     df_key.loc[df_key['key'] == row['key']] = 'max_' + str(row['key'])
        # print(df_key)
        # for model in self.models:
        #     predicted_dfs[model_check].loc[ predicted_dfs[model_check]['key'] == [row['key']]]
        list_pred = list()
        for index, row in df_key.iterrows():
            model_check = self.models[0]
            while(True):
                y_pred_df = predicted_dfs[model_check]
                if int(y_pred_df.loc[y_pred_df['key'] == row['key']]['y_pred'].iloc[0]) == 1:
                    next_step = model_check.type_1
                else:
                    next_step = model_check.type_0
                if len(next_step) == 1:
                    list_pred.append(int(next_step[0]))
                    break
                else:
                    try:
                        model_check = [x for x in self.models if sorted(x.all_cat_tested) == sorted(next_step)][0]
                    except Exception as e:
                        print(e)
                        raise e
                        break

        df_key['y_pred'] = list_pred
        self.predicted_df= df_key[['key','y_pred']]
        return self.predicted_df


        # merged_df = slice_X1_test.merge(slice_x2_data, on='key', how='left').merge(slice_x3_data, on='key', how='left')