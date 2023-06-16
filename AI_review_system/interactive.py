"""
File: interactive.py
Name: Jason Wu
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""
from util import *
from submission import *
FilePath = '/Users/yuchenwu/Documents/Mac14/SC201/Assignment/SC201Assignment2/weights'


def main():
	'''
	input: a sentence or words
	output:  the weights of each words.
	'''
	featureExtractor = extractWordFeatures
	weights = read_data()
	interactivePrompt(featureExtractor, weights)


def read_data():
	'''
	read the file and store in a dictionary.
	@return: weights, which is made in submission.py
	'''
	weights = {}
	with open (FilePath,'r') as f:
		for line in f:
			line_lst = line.split()
			weights[line_lst[0]] = float(line_lst[1])
	return weights


if __name__ == '__main__':
	main()