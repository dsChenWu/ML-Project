"""
File: titanic_level2.py
Name: Jason
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)

	################################
	data.pop('PassengerId')
	data.pop('Name')
	data.pop('Ticket')
	data.pop('Cabin')
	#print(data)

												# for i in range(len(data.Sex)):
												# 	if data.Age[i] or data.Embarked[i] == '':
												# 		data = data.drop(i,axis=0)
	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2
	# nan_cache['Age'] = Age_mean
	# nan_cache['Embarked'] = Emabrked_mean

	################################
	if mode == 'Train':
		data = data.dropna()
		labels = data.pop('Survived')
		return data, labels
	elif mode == 'Test':
		Age_mean = round(training_data.Age.mean(), 3)
		Emabrked_mean = round(training_data.Embarked.mean(), 3)
		Fare_mean = round(training_data.Fare.mean(), 3)
		data['Age'].fillna(Age_mean, inplace= True)
		data['Embarked'].fillna(Emabrked_mean, inplace=True)
		data['Fare'].fillna(Fare_mean, inplace=True)
		return data


def one_hot_encoding(data):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	############################
	# Female: Sex_0
	data['Sex_0'] = 0
	data.loc[data.Sex == 0, 'Sex_0'] = 1
	# Male: Sex_1
	data['Sex_1'] = 0
	data.loc[data.Sex == 1, 'Sex_1'] = 1
	# Pclass_0
	data['Pclass_0'] = 0
	data.loc[data.Pclass == 1, 'Pclass_0'] = 1
	# Pclass_1
	data['Pclass_1'] = 0
	data.loc[data.Pclass == 2, 'Pclass_1'] = 1
	# Pclass_2
	data['Pclass_2'] = 0
	data.loc[data.Pclass == 3, 'Pclass_2'] = 1
	# Embarked_0
	data['Embarked_0'] = 0
	data.loc[data.Embarked == 0, 'Embarked_0'] = 1
	data['Embarked_1'] = 0
	data.loc[data.Embarked == 1, 'Embarked_1'] = 1
	data['Embarked_2'] = 0
	data.loc[data.Embarked == 2, 'Embarked_2'] = 1
	data.pop('Sex')
	data.pop('Pclass')
	data.pop('Embarked')
	############################
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	############################
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)
	############################
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.8019662921348315
	TODO: real accuracy on degree2 -> 0.8370786516853933
	TODO: real accuracy on degree3 -> 0.8764044943820225
	"""
	data, labels = data_preprocess(TRAIN_FILE, mode='Train', training_data=None)
	data = one_hot_encoding(data)
	standardizer = preprocessing.StandardScaler()
	X = standardizer.fit_transform(data)

	# degree 2
	poly_phi_2 = preprocessing.PolynomialFeatures(degree=2)
	X_poly_2 = poly_phi_2.fit_transform(X)

	#degree 3
	poly_phi_3 = preprocessing.PolynomialFeatures(degree=3)
	X_poly_3 = poly_phi_3.fit_transform(X)

	h = linear_model.LogisticRegression(max_iter=1000)
	classifier_1 = h.fit(X,labels)
	train_acc_1 = classifier_1.score(X, labels)
	print('Train Acc 1: ', train_acc_1)

	classifier_2 = h.fit(X_poly_2,labels)
	train_acc_2 = classifier_2.score(X_poly_2, labels)
	print('Train Acc 2: ', train_acc_2)

	classifier_3 = h.fit(X_poly_3, labels)
	train_acc_3 = classifier_3.score(X_poly_3, labels)
	print('Train Acc 3: ', train_acc_3)


if __name__ == '__main__':
	main()
