'''
Main with automated evaluation on a test set
'''

import data_processing
import re, math, operator
import sys
import evaluate3
import pandas as pd 
import numpy as np

# upload data
print "loading data...\n"
print "loading hardcoded_question_set...\n"
dataProcessor = data_processing.DataProcessor()
path_train = '../../data2/'
file_name_train = 'test2.txt' 
hardcoded_question_set = dataProcessor.load_data(path_train, file_name_train)
print "loading test set...\n"
path_test = '../../data2/'
file_name_test = 'test_set2.tsv' #'test_set2.tsv'
test_set = dataProcessor.load_test_data(path_test, file_name_test)#.iloc[[6]]


# predict
evaluator = evaluate3.Evaluator3(hardcoded_question_set)
print "predicting...\n"
predictions, scores = evaluator.predict(test_set, hardcoded_question_set, projection_matrix=None, mult_answ = True)
print "evaluation starts...\n"
results = evaluator.merge_test_set_with_predictions(test_set, predictions, scores)
accuracy = evaluator.compute_accuracy(results)
print results
print "accuracy:"
print accuracy
print "evaluations ends...\n"


# write results in a tsv file
with open ('../../Data2/results3.tsv', 'w') as test_set:
	# First, write hardcoded questions (and modify manually user questions)
	for hardcoded_question, user_question, prediction, score, correct in zip(
			results['hardcoded_question'], 
			results['user_question'],
			results['prediction'],
			results['scores'],
			results['correct']):
		test_set.write("{}\t{}\t{}\t{}\t{}\n".format(hardcoded_question, user_question, prediction, score, correct))

