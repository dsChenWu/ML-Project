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
import random


TRAINING_FILE = 'boston_housing/train.csv'
TESTING_FILE = 'boston_housing/test.csv'


def main():
    # result = 0
    # for i in range(200):
    #Split data
    x_train, x_val, y_train, y_val = split_data(TRAINING_FILE, mode='Train')
    x_test, ID_ind = split_data(TESTING_FILE, mode='Test')

    # Data preprocessing
    x_train_scalerd, x_val_scalerd, x_test_scalerd = data_preprocessing(x_train, x_val, x_test, transform= 'Standardization')

    #Poly_feature
    poly_phi = preprocessing.PolynomialFeatures(degree=2)
    x_train_scalerd_poly = poly_phi.fit_transform(x_train_scalerd)
    x_val_scalerd_poly = poly_phi.transform(x_val_scalerd)
    x_test_scalerd_poly = poly_phi.transform(x_test_scalerd)

    # Start training
    h = ensemble.RandomForestRegressor(n_estimators=50, min_samples_leaf=1, max_features=0.5714115800622069)
    regressor = h.fit(x_train_scalerd_poly, y_train)

    y_pred_train = regressor.predict(x_train_scalerd_poly)
    y_pred_val = regressor.predict(x_val_scalerd_poly)

    r_score_train = regressor.score(x_train_scalerd_poly, y_train)
    r_score_val = regressor.score(x_val_scalerd_poly, y_val)

    rmse_train = metrics.mean_squared_error(y_train, y_pred_train, squared=False)
    rmse_val = metrics.mean_squared_error(y_val, y_pred_val, squared=False)
    # result += rmse_val
    print('R_square_train: ', r_score_train, '\n', 'RMSE_train: ', rmse_train)
    print('R_square_val: ', r_score_val, '\n', 'RMSE_val: ', rmse_val)
    # print('AVG: ',result/200)
    y_pred_test = regressor.predict(x_test_scalerd_poly)
    out_file(y_pred_test, 'st2_degree_2_random_forest_no_outliers.csv', ID_ind)

      # {'n_estimators': 50, 'min_samples_leaf': 1, 'max_leaf_nodes': None, 'max_features': 0.5714115800622069, 'max_depth': None} 0.8204494242031963
      # {'n_estimators': 98, 'min_samples_leaf': 1, 'max_depth': 14} 0.8111564851324535


def split_data(file, mode='Train'):
    if mode == 'Train':
        data = pd.read_csv(file)
        data = data.drop('ID', axis=1)
        #print(data.isna().sum().sort_values())  # Check if there are missing values.
        for i in data.columns:
            fifth_perc = np.percentile(data[i],5)  # Get the 5%th of the value
            nighty_fifth_perc = np.percentile(data[i], 95)  # Get the 95%th of the value
            data[i] = np.where(data[i]<fifth_perc, fifth_perc,data[i])  # Percentile Based Flooring and Capping
            data[i] = np.where(data[i]>nighty_fifth_perc, nighty_fifth_perc, data[i])  #Percentile Based Flooring and Capping

        y = data.pop('medv')

        # strategy two
        feature_names_2 = ['crim','zn', 'nox', 'rm', 'tax', 'ptratio','black','lstat']
        data = data[feature_names_2]

        x_train, x_val, y_train, y_val = model_selection.train_test_split(data, y, test_size=0.4, random_state=5)
        return x_train, x_val, y_train, y_val
    else:
        data = pd.read_csv(file)
        ID_ind = data.pop('ID')

        # strategy two
        feature_names_2 = ['crim', 'zn', 'nox', 'rm', 'tax', 'ptratio', 'black', 'lstat']
        data = data[feature_names_2]

        #print(data.isna().sum().sort_values())  # Check if there are missing values.
        return data, ID_ind


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


def out_file(predictions, filename, ID_ind):
    """
    : param predictions: numpy.array, a list-like data structure that stores 0's and 1's
    : param filename: str, the filename you would like to write the results to
    """
    print('\n===============================================')
    print(f'Writing predictions to --> {filename}')
    with open(filename, 'w') as out:
        out.write('ID,medv\n')
        start_id = 0
        for ans in predictions:
            out.write(str(ID_ind[start_id]) + ',' + str(ans) + '\n')
            start_id += 1
    print('===============================================')




if __name__ == '__main__':
    main()