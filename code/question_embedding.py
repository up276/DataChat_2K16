'''
Embedding questions using word2vec + CBOW model
Need to download gensim and textblob package and download Google pretrained word2vec embeddings

inspired from https://github.com/Poyuli/sentiment.analysis/blob/master/sentiment.py

Further potential improvements leads to explore: 
- train word2vec model on top of pre-trained Google word2vec
- Explore glove (standford equivalent for word2vec)
- Use characters N-grams
- Try skip-gram instead of CBOW (slower but better performance for sentimental analysis, what about similarity use case ?)
'''

from __future__ import division, unicode_literals
from textblob import TextBlob as tb
import numpy as np
import fasttext
import gensim
import pickle
# import fasttext

class TextEmbedder:
	'''
	embed text using word2vec + CBOW model
	'''

	def __init__(self,model):
		"""
		Initialize the TextEmbedder instance by loading Google pretrained word2vec embeddings
		"""
		self.user_selected_model = model
		print('loading words embeddings using ', self.user_selected_model)
		if self.user_selected_model=='fasttext':
			self.model = fasttext.load_model('/home/urjit/Capstone_2K16/fastText/fasttext_model.bin')
			self.model_vocab = self.model.words
			print "Fasttext model loaded !!!\n"
		if self.user_selected_model=='google':
			self.model = gensim.models.Word2Vec.load_word2vec_format('../../model/GoogleNews-vectors-negative300.bin', binary=True)
			self.model_vocab = self.model.vocab 
			print "Google word2vec model loaded !!!\n"		
		self.num_features = 300 # 300 if using word2vec (word2vec embedding dimension)
		self.new_words = {}


	def tf(self, word, blob):
		return blob.words.count(word) / float(len(blob.words))

	def tfidf(self, word, blob, idf_dic):
		return idf_dic[word] / self.tf(word, blob)


	def text_to_vec(self, question, idf_dictionnary):
		"""
		Generate a feature vector from a question, using word2vec + CBOW as embedding

		Input:
		    question: the question, as a list of words, to embed 
		    model: trained word2vec model
		    num_features: dimension of word2vec vectors
		    
		Output:
		    a numpy array representing the question (as a vector of dimension num_features since CBOW is performed)
		"""

		feature_vec = np.zeros((self.num_features), dtype="float32")
		words_embedded = 0
                #print "WORD EMBEDDING FOR CAR  : ",self.model['car'], " "
		for word in question:
			try:    
				#print " inside embedding loop mark1 "
				if word in idf_dictionnary.keys():
					word_tfidf = self.tfidf(word, tb(' '.join(question)), idf_dictionnary)
				else:
					word_tfidf = 1
				#print " inside embedding loop mark2 "
				#print "self.model[word]",type(self.model[word]), " : " ,self.model[word]
				#print "float(word_tfidf)",type(float(word_tfidf)), " : " ,float(word_tfidf)
				# print word," checking ...\n"
				if self.user_selected_model=='fasttext':
					# FASTTEXT BLOCK
					if word not in self.model_vocab:
						print word," not in vocab--------------------------------------------->\n"
						with open('../../data/unk_words.model', 'rb') as handle:
    							unknown_words_dict = pickle.load(handle)
                                                #print "unk model loaded !!!"
						if word not in unknown_words_dict:
							word_vec = np.random.uniform(-1.0, 1.0,300)
							print "word embedding :",word_embedding
							unknown_words_dict[word] =  word_embedding
							print "word embedding added in dict "
                                                	with open('../../data/unk_words.model', 'wb') as handle: 
    								pickle.dump(unknown_words_dict,handle)
							print "dict saved back as model"
							unk_list = open('fasttext_unknown_words.txt','a')
				                	print >>unk_list,word #,",",val #p_id_data['st_winlose'].unique()
				                	unk_list.close()
						else:
                                                        print 'word_vec in unkwords'
							word_vec = unknown_words_dict[word]
					else:
						word_vec = self.model[word]
				        a1 = [x*float(word_tfidf) for x in word_vec]
		                        feature_vec += a1
 
				if self.user_selected_model=='google':
					# GOOGLE WORD2VEEC BLOCK
					if word not in self.model_vocab:
						#print word," not in vocab--------------------------------------------->\n"
						with open('../../data/unk_words.model', 'rb') as handle:
							unknown_words_dict = pickle.load(handle)
						if word not in unknown_words_dict:
							word_vec = np.random.uniform(-1, 1, 300)
							print word, word_vec[:10]
							unknown_words_dict[word] =  word_vec
							print "word embedding added in dict "
							with open('../../data/unk_words.model', 'wb') as handle: 
								pickle.dump(unknown_words_dict,handle)
							print "dict saved back as model"
							# unk_list = open('../../google_unknown_words.txt','a')
							# print >>unk_list,word #,",",val #p_id_data['st_winlose'].unique()
							# unk_list.close()
						else:
							word_vec = unknown_words_dict[word]
					else:
						word_vec = self.model[word]
					#feature_vec += self.model[word] * float(word_tfidf)
					feature_vec += word_vec * float(word_tfidf)

				#print " inside embedding loop mark3 "
				#unk_list = open('fasttext_unknown_words.txt','a')
                                #print >>unk_list," Worf found in dictionary : ",word #,",",val #p_id_data['st_winlose'].unique()
                                #unk_list.close()
				words_embedded += 1
                                # print word," checked ...\n"
		                #print " inside embedding loop "
			except:
                                #if self.user_selected_model=='fasttext':	
                                #unk_list = open('fasttext_unknown_words.txt','a')
                                #print >>unk_list," ",word #,",",val #p_id_data['st_winlose'].unique()
                                #unk_list.close()
				continue 
		feature_vec /= float(words_embedded)
		return feature_vec


	def text_to_matrix(self, questions, idf_dictionnary):
		"""
		Generate a m-by-n numpy array from all questions, where m is number of questions, 
		and n is num_feature

		Input:
		        questions: a pandas data frame containingg the questions. 
		        model: trained word2vec model
		        num_feature: dimension of word2vec vectors
		Output: m-by-n numpy array, where m is len(questions) and n is num_feature
		"""

		review_feature_vecs = np.zeros((questions.shape[0], self.num_features), dtype="float32")

		i = 0
		for question in questions['hardcoded_question']:
			review_feature_vecs[i] = self.text_to_vec(question.strip().split(), idf_dictionnary)
			i += 1
		   
		return review_feature_vecs
