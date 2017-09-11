'''
Script to build the test set to evaluate framework.
This set is a .tsv file with the format 'true question' <tab> 'user question' 
It is made of:
	- Hardcoded questions : true hardcoded question <tab> user question --> to write manually 
	- 10 most similar questions from pool of questions per hardcoded question (to make 
		task more difficult for the model) : 'fallback' <tab> user fallback question
'''

import question_embedding
import similarity
import data_processing
import re, math, operator
import sys
import evaluate
import pandas as pd 
import numpy as np

# upload data
print "loading data...\n"
dataProcessor = data_processing.DataProcessor()
path_train = '../../Data2/'
file_name_train = 'train.txt' 
train_set = dataProcessor.load_datasetbuilding_data(path_train, file_name_train)
path_hardcoded_questions = '../../Data2/'
file_name_hardcoded_questions = 'test2.txt' 
hardcoded_question_set = dataProcessor.load_data(path_hardcoded_questions, file_name_hardcoded_questions)

# build idf dictionnary
idf_dictionnary = dataProcessor.build_sklearn_idf_dic(train_set)

# embed questions
model = 'google'
textEmbedder = question_embedding.TextEmbedder(model)
print "embedding questions...\n"
embedded_train_set = textEmbedder.text_to_matrix(train_set, idf_dictionnary)

# get fallbacks questions similar to hardcoded questions
print "generating fallback questions...\n"
evaluator = evaluate.Evaluator()
cosineSim = similarity.Similarity()
fallback_questions = []
for question in hardcoded_question_set['hardcoded_question']:
	print " "
	print "user question:"
	print question
	question = question.split()
	embeded_question = textEmbedder.text_to_vec(question, idf_dictionnary)
	best_match, best_match_index = cosineSim.find_best_match(embedded_train_set, embeded_question, nmax=10)
	for index in best_match_index:
		fallback_questions.append(train_set['hardcoded_question'][index])
		print train_set['hardcoded_question'][index]
fallback = ['fallback' for i in range(len(fallback_questions))]

# create test.tsv file
# test.tsv has the format 'true question' <tab> 'user question'
with open ('../../Data2/test_new.tsv', 'w') as test_set:
	# First, write hardcoded questions (and modify manually user questions)
	for hardcoded_question, hardcoded_question in zip(hardcoded_question_set['hardcoded_question'], 
		hardcoded_question_set['hardcoded_question']):
		test_set.write("{}\t{}\n".format(hardcoded_question, hardcoded_question))
	# Then write fallback questions
	for f, fq in zip(fallback, fallback_questions):
		test_set.write("{}\t{}\n".format(f, fq))

