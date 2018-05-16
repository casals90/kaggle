#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


TRAINIG_DATA = "data/train.csv"
TEST_DATA = "data/test.csv"

############
### Main ###
############
    
if __name__ == "__main__":
    df_train = pd.read_csv(TRAINIG_DATA, dtype={'Age': np.float64})
    df_test = pd.read_csv(TEST_DATA, dtype={'Age': np.float64})
    
    print(df_train.info())
    
    #####################
    ### Data cleaning ###
    #####################
    
    # print(df_train.head())
    
    # First we check the columns with some NaN
    # null_columns = df_train.columns[df_train.isnull().any()]
    # print(null_columns)
    
    # Change NaN to NO, and then create a new column with 0 or 1
    df_train['Cabin'].fillna("NO", inplace=True)
    df_train['Cabin01'] = df_train['Cabin'].apply(lambda x: 0 if x == "NO" else 1)
    
    # For age
    #mean_age = df_train['Age'].mean()
    #df_train['Age'].fillna(mean_age, inplace=True)
    
    # For Sex: Change male and famele for 0 and 1
    df_train['Sex01'] = df_train['Sex'].map({'female': 1, 'male': 0})
    
    # For name, adding a column as a name lenght
    df_train['NameLen'] = df_train['Name'].apply(lambda x: len(x))
    
    # For Embarked
    df_train['Embarked'].fillna("S", inplace=True)
    df_train['Embarked012'] = df_train['Embarked'].map({'C':0, 'Q':1, 'S':2})
    
    # Split X and y
    # 'Pclass', 'SibSp', 'Parch', 'Sex01', 'NameLen', 'Embarked012'
    X = df_train[['Sex01', 'Embarked012', 'Pclass']]
    y = df_train['Survived'].as_matrix()
    
    # Split the dataset in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)
    
    ###################
    ### Grid search ###
    ###################
    
    tuned_parameters = [{'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10, 100, 1000]}]
    scores = ['precision', 'recall']
    
    best_params = {}

    for score in scores:
        clf = GridSearchCV(svm.SVC(), 
                           tuned_parameters, 
                           cv=5,scoring='{}_macro'.format(score))
        clf.fit(X_train, y_train)

        print("Best parameters:")
        print(clf.best_params_)
		
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))

    C = clf.best_params_['C']
    kernel = clf.best_params_['kernel']
    
    print(C)
    print(kernel)
    
    ############
    ### SVM ###
    ###########
    
    clf_svm = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    print('#'*10)
    print('SVM:')
    print(clf_svm.score(X_test, y_test))
    
    #####################
    ### Random Forest ###
    #####################
    
    clf_random_forest = RandomForestClassifier(max_depth=10, random_state=42)
    clf_random_forest.fit(X_train, y_train)
    print('#'*10)
    print('Random Forest:')
    print(clf_random_forest.feature_importances_)
    print(clf_random_forest.score(X_test, y_test))
    
    
    ############
    ### Test ###
    ############
    
    # For sex
    df_test['Sex01'] = df_test['Sex'].map({'female': 1, 'male': 0})
    
    # For Embarked
    df_test['Embarked'].fillna("S", inplace=True)
    df_test['Embarked012'] = df_test['Embarked'].map({'C':0, 'Q':1, 'S':2})
    
    test = df_test[['Sex01', 'Embarked012', 'Pclass']]
    y_pred = clf_svm.predict(test)
    
    # Construc dataframe for data format kaggle evaluation
    
    final_columns = ["PassengerId", "Survived"]
    pass_id_test = df_test['PassengerId']
    
    data = {'PassengerId': df_test['PassengerId'],
            'Survived': y_pred}
    
    df_final = pd.DataFrame(data)
    
    df_final.to_csv("titanic_eval.csv",
				  header=True,
                    sep=',',
                    index=False,
                    encoding='utf-8')
    
    