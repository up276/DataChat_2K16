'''
Main with automated evaluation on a test set
'''

import question_embedding
import similarity
import data_processing
import re, math, operator
import sys
import evaluate
import time
import pandas as pd 
import numpy as np

# upload data

user_model = ''
def printProgramInfos():
		print "\n==========================================="
		print "WELCOME TO THE DATACHAT"
		print "===========================================\n"

def InitialUserOptions():
		print "\nYou are currently in the main menu. Please choose one of the options below\n \
		       1) press 1 : Fasttext Model \n \
		       2) press 2 : Google Word2Vec model  \n \
		       3) Enter 'quit' to exit from the program  "

def InitialUserInputLoop():
		"""
		    Main menu loop
		    Ask the user which action he/she wants to take and send the user to the appropriate option.
		    The loop is break (and the program ends) whenever the user types quit.
		"""
 		global user_model
		userInput = ""
                model_flag = False
		#try:
		while not model_flag:
			userInput = raw_input("\nPlease enter the Input: ")
			if userInput == "1":
			    #self.ChatbotInputLoop()
			    user_model = 'fasttext'
                            model_flag  = True
			elif userInput == "2":
			    user_model = 'google'
                            model_flag  = True
			    #self.test_classifier()
			    break
			elif userInput == "quit":
			    ExitProgram()
			else:
			    print "\nOops...Incorrect Input...Please enter correct Input !!!\n"
                 #except KeyboardInterrupt:
		#	print "quitting..."
		#	sys.exit()

def ExitProgram():
		time.sleep(2)
		print "Exiting from the program...See you soon !!!)"
		time.sleep(2)
		sys.exit()


if __name__ == "__main__":
	#chatbot = ChatBot()
	#chatbot.InitiateFlow()
        printProgramInfos()
        InitialUserOptions()
	InitialUserInputLoop()

	'''
	print "loading data...\n"
	dataProcessor = data_processing.DataProcessor()
	path_train = '../../data/'
	file_name_train = 'short_train.txt'  #global_dataset_without_duplicates.txt
	hardcoded_question_set = dataProcessor.load_data(path_train, file_name_train)
	path_test = '../../data/'
	file_name_test = 'test_set.tsv'
	test_set = dataProcessor.load_test_data(path_test, file_name_test)
	'''


	print "loading data...\n"
	dataProcessor = data_processing.DataProcessor()
	path_train = '../../data2/'
	file_name_train = 'test2.txt'
	hardcoded_question_set = dataProcessor.load_data(path_train, file_name_train)
	path_test = '../../data2/'
	file_name_test = 'test_set2.tsv'
	test_set = dataProcessor.load_test_data(path_test, file_name_test)
	path_idf = '../../data2/'
	file_name_idf = 'train.txt'
	idf_set = dataProcessor.load_data(path_idf, file_name_idf)

	idf_dictionnary = dataProcessor.build_sklearn_idf_dic(idf_set)

	# embed questions
	print "embedding questions using ",user_model, "...\n"
	textEmbedder = question_embedding.TextEmbedder(user_model)
	embedded_hardcoded_question_set = textEmbedder.text_to_matrix(hardcoded_question_set, idf_dictionnary)

	# predict
	print "predicting...\n"
	evaluator = evaluate.Evaluator()
	predictions = evaluator.predict(test_set, embedded_hardcoded_question_set, hardcoded_question_set, textEmbedder, idf_dictionnary)
	print "evaluation starts...\n"
	results = evaluator.merge_test_set_with_predictions(test_set, predictions)
	accuracy = evaluator.compute_accuracy(results)
	print results
	print "accuracy:"
	print accuracy
	print "evaluations ends...\n"



