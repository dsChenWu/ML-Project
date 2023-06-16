"""
File: boston_housing_competition.py
Name: WU YU CHEN
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import math
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics
from sklearn import linear_model
from sklearn import tree
from sklearn import decomposition
from sklearn import svm
from sklearn import ensemble
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

TRAINING_FILE = 'boston_housing/train.csv'
TESTING_FILE = 'boston_housing/test.csv'

# Linear Regression
# Random_forest
# Bagging
# SVM
# Do PCA first and do the model later



def main():

    #Split data
    x_train, x_val, y_train, y_val = split_data(TRAINING_FILE, mode='Train')
    x_test = split_data(TESTING_FILE, mode='Test')

    # Data preprocessing
    x_train_scalerd, x_val_scalerd, x_test_scalerd = data_preprocessing(x_train, x_val, x_test, transform= 'Standardization')

    # Poly_feature
    poly_phi = preprocessing.PolynomialFeatures(degree=2)
    x_train_scalerd_poly = poly_phi.fit_transform(x_train_scalerd)
    # # x_val_scalerd_poly = poly_phi.transform(x_val_scalerd)
    # # x_test_scalerd_poly = poly_phi.transform(x_test_scalerd)

    # PCA_transformation
    h = decomposition.PCA(n_components=15)
    x_train_scalerd_poly_pca = h.fit_transform(x_train_scalerd_poly)

    # Cross_validation --> choose better models
    models = {'Linear Regression': linear_model.LinearRegression(),
              'Random_forest':ensemble.RandomForestRegressor(max_depth= 16, min_samples_leaf= 5),
              'bagging': ensemble.BaggingRegressor(base_estimator= ensemble.RandomForestRegressor(max_depth= 16, min_samples_leaf= 5),n_estimators=100),
              'SVM': svm.SVR(kernel='rbf', C=30.0)}
    result = []
    for model in models.values():
        kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=5)
        cv_results = model_selection.cross_val_score(model, x_train_scalerd_poly_pca, y_train, cv=kf)
        result.append(cv_results)
    print(result)
    print(np.mean(result))
    plt.boxplot(result, labels= models.keys())
    plt.xlabel('Regression Models')
    plt.ylabel('Cross-validation Score')
    plt.title('Comparison of Regression Models')
    plt.show()


def split_data(file, mode='Train'):
    if mode == 'Train':
        data = pd.read_csv(file)
        data = data.drop('ID', axis=1)
        # print(data.isna().sum().sort_values())  # Check if there are missing values.
        y = data.pop('medv')

        # strategy one
        # feature_names = ['crim', 'nox', 'rm', 'tax', 'ptratio', 'lstat']
        # data = data[feature_names]

        # strategy two
        # feature_names_2 = ['crim','zn', 'nox', 'rm', 'tax', 'ptratio','black','lstat']
        # data = data[feature_names_2]

        # strategy three
        # feature_names_3 = ['crim', 'zn', 'indus', 'rm', 'rad', 'ptratio', 'black', 'lstat']
        # data = data[feature_names_3]

        # strategy four
        feature_names_4 = ['ptratio', 'rm', 'lstat']
        data = data[feature_names_4]

        x_train, x_val, y_train, y_val = model_selection.train_test_split(data, y, test_size=0.4, random_state=5)
        return x_train, x_val, y_train, y_val
    else:
        data = pd.read_csv(file)
        data = data.drop('ID', axis=1)

        # strategy one
        # feature_names = ['crim', 'nox', 'rm', 'tax', 'ptratio', 'lstat']
        # data = data[feature_names]

        # strategy two
        # feature_names_2 = ['crim', 'zn', 'nox', 'rm', 'tax', 'ptratio', 'black', 'lstat']
        # data = data[feature_names_2]

        # strategy three
        # feature_names_3 = ['crim', 'zn', 'indus', 'rm', 'rad', 'ptratio', 'black', 'lstat']
        # data = data[feature_names_3]

        # strategy four
        feature_names_4 = ['ptratio', 'rm', 'lstat']
        data = data[feature_names_4]

        # print(data.isna().sum().sort_values())  # Check if there are missing values.
        return data


def data_preprocessing(training_data, val_data,testing_data, transform = 'Standardization'):
    if transform == 'Standardization':
        standardizer = preprocessing.StandardScaler()
        std_train_data = standardizer.fit_transform(training_data)
        std_val_data = standardizer.transform(val_data)
        std_test_data = standardizer.transform(testing_data)
        return  std_train_data, std_val_data, std_test_data
    elif transform == 'Normalization':
        normalizer = preprocessing.MinMaxScaler()
        nor_train_data = normalizer.fit_transform(training_data)
        nor_val_data = normalizer.transform(val_data)
        nor_test_data = normalizer.transform(testing_data)
        return nor_train_data, nor_val_data, nor_test_data
    else:
        return training_data, val_data, testing_data


if __name__ == '__main__':
    main()