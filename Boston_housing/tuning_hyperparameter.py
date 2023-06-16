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

    #Split data
    x_train, x_val, y_train, y_val = split_data(TRAINING_FILE, mode='Train')
    x_test = split_data(TESTING_FILE, mode='Test')

    # Data preprocessing
    x_train_scalerd, x_val_scalerd, x_test_scalerd = data_preprocessing(x_train, x_val, x_test, transform= 'Standardization')
    kf = model_selection.KFold(n_splits=4, shuffle=True, random_state=5)

    # Poly_feature
    poly_phi = preprocessing.PolynomialFeatures(degree=2)
    x_train_scalerd_poly = poly_phi.fit_transform(x_train_scalerd)

    # PCA_transformation
    h = decomposition.PCA(n_components=15)
    x_train_scalerd_poly_pca = h.fit_transform(x_train_scalerd_poly)
    var_retained = sum(h.explained_variance_ratio_)
    print('Var retained: ', var_retained)

# RandomizedSearchCV:
    # Linear Regression: Lasso
    param_dist = {'alpha': np.linspace(0.001, 1, 60)}
    lasso = linear_model.Lasso()
    lasso_cv = model_selection.RandomizedSearchCV(lasso, param_distributions=param_dist, cv=kf, n_iter=50)
    lasso_cv.fit(x_train_scalerd_poly_pca, y_train)
    print('Lasso Regression: ', lasso_cv.best_params_, lasso_cv.best_score_)
    print('-'*100)
    # Result_1: {'alpha': 0.30515423728813557} 0.6305037659750647 (st_2)
    # Result_2: {'alpha': 0.13567966101694914} 0.8336073287915887 (all)
    # Result_3: {'alpha': 0.18652203389830507} 0.8047415250244925 (all_poly)
    # Result_4: {'alpha': 0.30515423728813557} 0.6314302066802456 (st_3)
    # Result_5: {'alpha': 0.11873220338983051} 0.8033717912952439 (st_3_poly)

    # Linear Regression: Ridge
    param_dist = {'alpha': np.linspace(0.001, 1, 20),
                  'solver': ['sag', 'lsqr', 'auto', 'cholesky', 'sparse_cg']}
    ridge = linear_model.Ridge()
    ridge_cv = model_selection.RandomizedSearchCV(ridge, param_distributions=param_dist, cv=kf, n_iter=50)
    ridge_cv.fit(x_train_scalerd_poly_pca, y_train)
    print('Ridge Regression: ', ridge_cv.best_params_, ridge_cv.best_score_)
    print('-' * 100)
    # Result: {'solver': 'auto', 'alpha': 1.0} 0.6066334566428986 (st_2)
    # Result_2: {'solver': 'sag', 'alpha': 0.9473736842105264} 0.7964904460665725 (all)
    # Result_3: {'solver': 'sag', 'alpha': 0.5789894736842105} 0.8166963579767066 (all_poly)
    # Result_4: {'solver': 'lsqr', 'alpha': 1.0} 0.6110514351287175 (st_3)
    # Result_5: {'solver': 'sag', 'alpha': 0.7894947368421052} 0.793224594608065 (st_3_poly)

    # Random_Forest
    param_dist = {'max_depth': np.arange(5,20,1),'min_samples_leaf': np.arange(5,20,1),'n_estimators': np.arange(1,100)}
    random_forest = ensemble.RandomForestRegressor()
    random_forest_cv = model_selection.RandomizedSearchCV(random_forest, param_distributions=param_dist, cv=kf, n_iter=50)
    random_forest_cv.fit(x_train_scalerd_poly_pca, y_train)
    print('Random Forest: ', random_forest_cv.best_params_, random_forest_cv.best_score_)
    print('-' * 100)
    # Result_1: {'n_estimators': 50, 'min_samples_leaf': 1, 'max_leaf_nodes': None, 'max_features': 0.5714115800622069, 'max_depth': None} 0.8204494242031963 (st_2)
    # Result_2: {'n_estimators': 93, 'min_samples_leaf': 5, 'max_depth': 15} 0.8064073373612552 (all)
    # Result_3: {'n_estimators': 94, 'min_samples_leaf': 5, 'max_depth': 17} 0.80318634078078 (all_poly)
    # Result_4: {'n_estimators': 19, 'min_samples_leaf': 5, 'max_depth': 19} 0.7862899959835405 (st_3)
    # Result_5: {'n_estimators': 95, 'min_samples_leaf': 5, 'max_depth': 9} 0.7853231286972548 (st_3_poly)

    # Bagging: Bootstrap
    param_dist = {'base_estimator__max_depth': np.arange(1, 21),'base_estimator__min_samples_leaf': np.arange(1, 21)}
    bagging = ensemble.BaggingRegressor(base_estimator=ensemble.RandomForestRegressor())
    bagging_cv = model_selection.RandomizedSearchCV(bagging, param_distributions=param_dist, cv = kf, n_iter=50)
    bagging_cv.fit(x_train_scalerd_poly_pca, y_train)
    print('Bootstrap: ', bagging_cv.best_params_, bagging_cv.best_score_)
    print('-'*100)
    # Result_1: {'base_estimator__min_samples_leaf': 1, 'base_estimator__max_depth': 17} 0.8117923671652771 (st_2)
    # Result_2: {'base_estimator__min_samples_leaf': 2, 'base_estimator__max_depth': 14} 0.8091957297969522 (all)
    # Result_3: {'base_estimator__min_samples_leaf': 3, 'base_estimator__max_depth': 20} 0.8132627296846509 (all_poly)
    # Result_4: {'base_estimator__min_samples_leaf': 2, 'base_estimator__max_depth': 18} 0.8020152327626164 (st_3)
    # Result_5: {'base_estimator__min_samples_leaf': 1, 'base_estimator__max_depth': 18} 0.8022486317638562 (st_3_poly)

    # SVM
    param_dist= {'kernel': ['linear', 'poly', 'rbf'], 'C': np.linspace(0.1,30)}
    svm_regressor = svm.SVR()
    svm_regressor_cv = model_selection.RandomizedSearchCV(svm_regressor, param_distributions=param_dist, cv=kf, n_iter=50)
    svm_regressor_cv.fit(x_train_scalerd_poly_pca, y_train)
    print('SVM: ', svm_regressor_cv.best_params_, svm_regressor_cv.best_score_)
    # Result_2: {'kernel': 'linear', 'C': 0.1} 0.8320116871031518 (all)
    # Result_3: kernel='rbf', C=25.728571428571428 (all_poly)
    # Result_4: {'kernel': 'rbf', 'C': 20.846938775510203} 0.7931608322017141 (st_3)
    # Result_5: {'kernel': 'rbf', 'C': 30.0} 0.8009293275660782 (st_3_poly)

# GridSearchCV
    # Linear Regression: Lasso
    param_grid = {'alpha': np.linspace(0.001, 1, 60)}
    lasso = linear_model.Lasso()
    lasso_cv = model_selection.GridSearchCV(lasso, param_grid=param_grid, cv=kf)
    lasso_cv.fit(x_train_scalerd, y_train)
    print('Lasso Regression: ', lasso_cv.best_params_, lasso_cv.best_score_)  # {'alpha': 0.13567966101694914} 0.6623163496610756
    print('-'*100)

    # Linear Regression: Ridge
    param_grid = {'alpha': np.linspace(0.001, 1, 20), 'solver': ['sag', 'lsqr', 'auto', 'cholesky', 'sparse_cg']}
    ridge = linear_model.Ridge()
    ridge_cv = model_selection.GridSearchCV(ridge, param_grid=param_grid, cv=kf)
    ridge_cv.fit(x_train_scalerd, y_train)
    print('Ridge Regression: ', ridge_cv.best_params_, ridge_cv.best_score_)  # {'alpha': 1.0, 'solver': 'auto'} 0.6554616184009314
    print('-' * 100)

    # Random_Forest
    param_grid = {'max_depth': np.arange(5,20,1),'min_samples_leaf': np.arange(5,20,1)}
    random_forest = ensemble.RandomForestRegressor()
    random_forest_cv = model_selection.GridSearchCV(random_forest, param_grid = param_grid, cv=kf)
    random_forest_cv.fit(x_train_scalerd, y_train)
    print('Random Forest: ', random_forest_cv.best_params_, random_forest_cv.best_score_)  # {'max_depth': 10, 'min_samples_leaf': 5} 0.7934094718619092
    print('-' * 100)
    ensemble.RandomForestClassifier
    # Result_6: {'max_depth': 16, 'min_samples_leaf': 5} 0.8051928936142355 (st_2_poly_pca)

    # Bagging: Bootstrap
    param_grid = {'base_estimator__max_depth': np.arange(1, 21),'base_estimator__min_samples_leaf': np.arange(1, 21)}
    bagging = ensemble.BaggingRegressor(base_estimator=ensemble.RandomForestRegressor())
    bagging_cv = model_selection.GridSearchCV(bagging, param_grid=param_grid, cv = kf)
    bagging_cv.fit(x_train_scalerd, y_train)
    print('Bootstrap: ', bagging_cv.best_params_, bagging_cv.best_score_)  # {'base_estimator__max_depth': 11, 'base_estimator__min_samples_leaf': 1} 0.8298007940755733
    print('-'*100)

    # SVM
    param_grid= {'kernel': ['linear', 'poly', 'rbf'], 'C': np.linspace(0.1,30)}
    svm_regressor = svm.SVR()
    svm_regressor_cv = model_selection.GridSearchCV(svm_regressor, param_grid = param_grid, cv=kf)
    svm_regressor_cv.fit(x_train_scalerd, y_train)
    print('SVM: ', svm_regressor_cv.best_params_, svm_regressor_cv.best_score_)  # {'C': 30.0, 'kernel': 'rbf'} 0.8148821767670287


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
        # feature_names_3 = ['crim','zn','indus', 'rm', 'rad', 'ptratio','black','lstat']
        # data = data[feature_names_3]

        # strategy four
        feature_names_4 = ['ptratio', 'rm', 'lstat']
        data = data[feature_names_4]

        x_train, x_val, y_train, y_val = model_selection.train_test_split(data, y, test_size=0.4, random_state=5)
        return x_train, x_val, y_train, y_val
    else:
        data = pd.read_csv(file)
        data = data.drop('ID', axis=1)

        # # strategy one
        # feature_names = ['crim', 'nox', 'rm', 'tax', 'ptratio', 'lstat']
        # data = data[feature_names]

        # strategy two
        feature_names_2 = ['crim', 'zn', 'nox', 'rm', 'tax', 'ptratio', 'black', 'lstat']
        data = data[feature_names_2]

        # strategy three
        # feature_names_3 = ['crim','zn','indus', 'rm', 'rad', 'ptratio','black','lstat']
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