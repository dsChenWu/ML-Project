#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    d = defaultdict(int)
    lst_split = x.split()
    for i in lst_split:
        d[i] += 1
    return d
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    #trainExamples is a list that storing lot of tuple.

    # Have initial weights
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    for epoch in range(numEpochs):
        def predictor(x_ex): # make str in the function
            phi_vector = featureExtractor(x_ex) # Get tge featureVector
            k = dotProduct(phi_vector, weights) #
            y = 1 if k >0 else -1
            return y
        cost = 0
    # Calculate k = weights * feature_vector(i)
        for x,y in trainExamples:
            phi_vector = featureExtractor(x)
            k = dotProduct(phi_vector, weights)
            # Calculate h = sigmoid(k)
            h = 1/ (1+math.exp(-k))
            # Calculate Loss function = -(y*math.log(h)+(1-y)*math.log(1-h))
            if y == 1:
                loss_function = -(y*math.log(h))
            else:
                y = 0
                loss_function = -(math.log(1-h))
            cost += loss_function # Calculate Cost function
            # G.D
            learning_rate = float(-1*alpha*(h-y))
            increment(weights,learning_rate, phi_vector)  # w = w - alpha*((h-y)*x #increment has done the for loop
        # print the training & validation error for each epoch
        print('Training Error: ','(',str(epoch),'epoch','): ',evaluatePredictor(trainExamples,predictor))
        print('Validation Error: ','(',str(epoch),'epoch','): ',evaluatePredictor(validationExamples, predictor))

        ''' Notes:
        for j in range(len(phi_vector)):
            weights[j] = increment(weights, learning_rate, phi_vector)

            else:  # Add
                for x,y in validationExamples:
                    phi_vector = featureExtractor(x)
                    k = dotProduct(phi_vector, weights)
        '''
    # END_YOUR_CODE#
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    numExamples: Generate the number of reply
    weights: Store all {key: value}
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        phi = {}
        num = 0
        numExamples = random.randint(0, len(weights)) # Get the number of example from weights
        # Create phi_vector which is the subset of weights
        for key, value in weights.items():
            phi[key] = value
            if num < numExamples:
                num += 1
            else:
                break
        # Get the dotproudct of phi_vector & weights
        k = dotProduct(phi, weights)
        y = 1 if k>0 else 0
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        dic = defaultdict(int)
        new_x = ''
        for i in range(len(x)): # Use this for loop to make sure the string has no '', space.
            if x[i].isalpha():
                new_x += x[i]
        for j in range(len(new_x)-n+1): # get the len(n) of str and store in dictionary
            sub = new_x[j:j+n]
            dic[sub] += 1
        return dic
        # END_YOUR_CODE
    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

