'''
Main
'''

import question_embedding
import similarity
import data_processing
import re, math, operator
import sys

# upload data
dataProcessor = data_processing.DataProcessor()
#path = 'data'
#file_name = 'gloabl_question_set.csv'
#questions = dataProcessor.load_data(path, file_name)

path = '../../data/'
file_name = 'short_train.txt'  #global_dataset_without_duplicates.txt
questions = dataProcessor.load_data(path, file_name)

#path_test = '../data/'
#file_name_test = 'test_set.tsv'
#test_set = dataProcessor.load_test_data(path_test, file_name_test)

# build hardcoded questions data set dictionnary
dictionnary = dataProcessor.build_dictionnary(path, file_name)
idf_dictionnary = dataProcessor.build_sklearn_idf_dic(questions)

textEmbedder = question_embedding.TextEmbedder()
# idf_dic = textEmbedder.build_idf_dic(dictionnary, questions)

# embed questions set 
embedded_questions = textEmbedder.text_to_matrix(questions, dictionnary)

def find_best_match(question):
	# embed user's question
	question =  dataProcessor.clean_str(question)
	embedded_question = textEmbedder.text_to_vec(question, dictionnary)

	# perform cosine similarity and find best match
	cosineSim = similarity.Similarity()
	embed_best_match, best_match_index = cosineSim.find_best_match(embedded_questions,embedded_question)
	best_match = questions[best_match_index]

	return best_match

userInput = ""
try:
	while userInput != "quit":
		userInput = raw_input("\nPlease ask your question : ")
		best_match = find_best_match(userInput)
		print('user question :')
		print(userInput)
		print('the model finds this is the closest question in database :')
		print(best_match)
except KeyboardInterrupt:
	print "quitting..."
	sys.exit()

