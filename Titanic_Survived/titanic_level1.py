"""
File: titanic_level1.py
Name: 
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

DATA = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

import math
import statistics
import util
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None): #Why mode and training cannot modify
	"""
	:param filename: str, the filename to be processed
	:param data: an empty Python dictionary
	:param mode: str, indicating if it is training mode or testing mode
	:param training_data: dict[str: list], key is the column name, value is its data
						  (You will only use this when mode == 'Test')
	:return data: dict[str: list], key is the column name, value is its data
	"""
	############################
	with open(filename, 'r') as f:
		first_line = True
		for line in f:
			line = line.strip().split(",")
			if first_line:  # If first line, data should store the key
				first_line = False
				if mode == 'Train':
					for i in range(len(line)):
						if line[i] in DATA:
							data[line[i]] = []
				else:
					for i in range(1,len(line)):
						if line[i] in DATA:
							data[line[i]] = []
			else:  # if not first line, data should store the value.
				if mode == 'Train':
					if line[6] and line[12]:  # Why cannot use line[6] or line[12] != ''
						for j in range(len(line)):
							if j == 1:  # Survived
								data['Survived'].append(int(line[j]))
							elif j == 2:  # Pclass
								data['Pclass'].append(int(line[j]))
							elif j == 5:  # Sex
								if line[j] == 'male':
									data['Sex'].append(1)
								else:
									data['Sex'].append(0)
							elif j == 6:  # Age
								#if line[j] != '':
								data['Age'].append(float(line[j]))
							elif j == 7:  # Sibsp
								data['SibSp'].append(int(line[j]))
							elif j == 8:  # Parch
								data['Parch'].append(int(line[j]))
							elif j == 10:  # Fare
								data['Fare'].append(float(line[j]))
							elif j == 12:  # Embarked
								if line[j] == "S":
									data['Embarked'].append(0)
								elif line[j] == "C":
									data['Embarked'].append(1)
								elif line[j] == "Q":
									data['Embarked'].append(2)
				else:
					#print(data)
					for j in range(1,len(line)):
						if j == 1:  # Pclass
							if line[j].isdigit():
								data['Pclass'].append(int(line[j]))
							else:
								data['Pclass'].append(round(float(statistics.mean(training_data['Pclass'])),3))
						elif j == 4:  # Sex
							if line[j] == 'male':
								data['Sex'].append(1)
							elif line[j]== 'female':
								data['Sex'].append(0)
							else:
								data['Sex'].append(round(float(statistics.mean(training_data['Sex'])),3))
						elif j == 5:  # Age
							if line[j]:
								data['Age'].append(float(line[j]))
							else:
								data['Age'].append(round(float(statistics.mean(training_data['Age'])),3))
						elif j == 6:  # Sibsp
							if line[j].isdigit():
								data['SibSp'].append(int(line[j]))
							else:
								data['SibSp'].append(round(float(statistics.mean(training_data['SibSp'])),3))
						elif j == 7:  # Parch
							if line[j].isdigit():
								data['Parch'].append(int(line[j]))
							else:
								data['Parch'].append(round(float(statistics.mean(training_data['Parch'])),3))
						elif j == 9:  # Fare
							if line[j]:
								data['Fare'].append(float(line[j]))
							else:
								data['Fare'].append(round(float(statistics.mean(training_data['Fare'])),3))
						elif j == 11:  # Embarked
							if line[j] == "S":
								data['Embarked'].append(0)
							elif line[j] == "C":
								data['Embarked'].append(1)
							elif line[j] == "Q":
								data['Embarked'].append(2)
							else:
								data['Embarked'].append(round(float(statistics.mean(training_data['Embarked'])),3))
	############################
	return data


def one_hot_encoding(data: dict, feature: str):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: dict[str, list], remove the feature column and add its one-hot encoding features
	"""
	############################
	if feature == 'Pclass':
		data['Pclass_0'] = []
		data['Pclass_1'] = []
		data['Pclass_2'] = []
		for ele in data['Pclass']:
			if ele == 1:
				data['Pclass_0'].append(1)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(0)
			elif ele == 2:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(1)
				data['Pclass_2'].append(0)
			else:
				data['Pclass_0'].append(0)
				data['Pclass_1'].append(0)
				data['Pclass_2'].append(1)
		data.pop('Pclass')
	elif feature == 'Sex':
		data['Sex_1'] = []
		data['Sex_0'] = []
		for ele in data['Sex']:
			if ele == 1:
				data['Sex_1'].append(1)
				data['Sex_0'].append(0)
			else:
				data['Sex_1'].append(0)
				data['Sex_0'].append(1)
		data.pop('Sex')
	elif feature == 'Embarked':
		data['Embarked_0'] = []
		data['Embarked_1'] = []
		data['Embarked_2'] = []
		for ele in data['Embarked']:
			if ele == 0:
				data['Embarked_0'].append(1)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(0)
			elif ele == 1:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(1)
				data['Embarked_2'].append(0)
			else:
				data['Embarked_0'].append(0)
				data['Embarked_1'].append(0)
				data['Embarked_2'].append(1)
		data.pop('Embarked')



	############################
	return data


def normalize(data: dict):
	"""
	:param data: dict[str, list], key is the column name, value is its data
	:return data: dict[str, list], key is the column name, value is its normalized data
	"""
	############################
	#print(data)
	for x,y in data.items():
		lst = []
		for ele in y:
			normalized_value = float(((ele - float(min(y)))/(float(max(y))-float(min(y)))))
			lst.append(normalized_value)
		data[x] = lst
	############################
	return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
	"""
	:param inputs: dict[str, list], key is the column name, value is its data
	:param labels: list[int], indicating the true label for each data
	:param degree: int, degree of polynomial features
	:param num_epochs: int, the number of epochs for training
	:param alpha: float, known as step size or learning rate
	:return weights: dict[str, float], feature name and its weight
	"""
	# Step 1 : Initialize weights
	weights = {}  # feature => weight
	keys = list(inputs.keys())
	print(keys)
	if degree == 1:
		for i in range(len(keys)):
			weights[keys[i]] = 0
	elif degree == 2:
		for i in range(len(keys)):
			weights[keys[i]] = 0
		for i in range(len(keys)):
			for j in range(i, len(keys)):
				weights[keys[i] + keys[j]] = 0
	# Step 2 : Start training

	#index = 0
	#label = inputs.pop('Survived')
	#print(inputs)
	for epoch in range(num_epochs):
		for i in range(len(inputs['Age'])): # Each data run over called one epoch
			d1 = {}
			if degree == 1:
				for j in range(len(keys)):
					d1[keys[j]] = inputs[keys[j]][i]  # Use list or dict?
			elif degree ==2:
				values = list(inputs.values())
				#print(values)
				for z in range(len(keys)):
					d1[keys[z]] = inputs[keys[z]][i]
					for j in range(z, len(values)):
						d1[keys[z]+keys[j]] = inputs[keys[z]][i] * inputs[keys[j]][i]  # Example: 'Age' * 'Sibsp' --> Age[0] *Sibsp[0]
			y = labels[i]

			# Step 3 : Feature Extract
			k = util.dotProduct(d1,weights)
			#k = util.dotProduct(weights, d1)
			h = 1/(1+math.exp(-k))
			loss_func= -(y*math.log(h)+(1-y)*math.log(1-h))

			# Step 4 : Update weights
			util.increment(weights, -alpha*(h-y), d1)  #weights = weight -alpha* dJ/d_theta
			#util.increment(d1, -alpha * (h - y), weights)
	return weights
