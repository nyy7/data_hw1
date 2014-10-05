#!/usr/bin/python

import random
import urllib2
import numpy 
from nltk.tokenize import *
from nltk.corpus import *
from nltk.stem import *
import pip, sys, os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

# split text into chunks, each chunk has l_num lines
def GenerateData(filename, l_num):
        data = []

        # generate data
        f = open (filename)
        paragraph = ""
        line_num = 0

        for line in f:
                paragraph += line
                line_num += 1
                if line_num % l_num == 0:
                        data.append(paragraph)
                        paragraph = ""
        f.close()
        return data

# data process
def data_process(data):
        data_processed = []
        for d in data:
        # tokenize
                word_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
                text1_tk = word_tokenizer.tokenize(d)
        # lower
                text1_l = [w.lower() for w in text1_tk]
        # unicode
                text1_l_uni = [unicode(w, errors='ignore') for w in text1_l]
        # remove stopwords
                text1_sw = [w for w in text1_l_uni if w not in stopwords.words('english')]
        # stemming
                stemmer = PorterStemmer()
                text1_st = [stemmer.stem_word(w) for w in text1_sw]
                data_processed.append(" ".join(text1_st))
        return data_processed

#cross_validation function	
def cross_val_score(clf, data, target, k):
	shuffle_arr = []
	size = len(data)
	for i in range(size):
		shuffle_arr.append(i)
	scores = []
	for i in range(0, k):
		#generate shuffled train and test dataset
		data_train_raw = []
		data_test_raw = []
		target_train = []
		target_test = []
		# seperate shuffled train and test dataset
                random.shuffle(shuffle_arr)
                shuffle_train = shuffle_arr[:size - size/k]
                shuffle_test = shuffle_arr[size-size/k :]
                for j in shuffle_train:
                        data_train_raw.append(data_total[j])
                        target_train.append(target[j])
                for r in shuffle_test:
                        data_test_raw.append(data_total[r])
                        target_test.append(target[r])

		data_train = data_process(data_train_raw)
		data_test = data_process(data_test_raw)

		# transform array of string to counts
		count_vect = CountVectorizer()
		X_train_counts = count_vect.fit_transform(data_train)
		# transform counts to frequencies
		tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
		X_train_tf = tf_transformer.transform(X_train_counts)
		
		# feature selection
		select = SelectPercentile(chi2, percentile = 10)
		X_train_fs = select.fit_transform(X_train_tf, target_train)
							
		# train the model
		clf_train = clf.fit(X_train_fs, target_train)

		# test the model
		X_new_counts = count_vect.transform(data_test)
		X_new_tfidf = tf_transformer.transform(X_new_counts)
		X_new_fs = select.transform(X_new_tfidf)
		test_result = clf_train.predict(X_new_fs)
		scores.append(GetPrecisionRecallF1(test_result, target_test))
		#clf_score =  clf_train.score(X_new_fs, target_test)
		#scores.append(clf_score)
	return scores

def GetPrecisionRecallF1(predict_result, real_result):
	tp = fp = tn = fn = 0.0
	for i in range(len(predict_result)):
		if predict_result[i] == 0 and real_result[i] == 0:
			tn += 1	
		if predict_result[i] == 0 and real_result[i] == 1:
			fn += 1	
		if predict_result[i] == 1 and real_result[i] == 0:
			fp += 1	
		if predict_result[i] == 1 and real_result[i] == 1:
			tp += 1	
	prec = tp/(tp+fp)
	recall = tp/(tp+fn)
	f1_score = 2*(prec*recall)/(prec+recall)
	return (prec, recall, f1_score)

k = 10

for data_size in range(1, 10, 2):

	data_auth1 = GenerateData('pg1661.txt', data_size)
	data_auth2 = GenerateData('pg31100.txt', data_size)

	data_total = data_auth1 + data_auth2

#generate target array
	target = []
	for d in data_auth1:
        	target.append(0)
	for d in data_auth2:
        	target.append(1)

	# laplace smoothing
	clf = MultinomialNB()
	# Maximum likelihood relative frequency estimator
	#clf = MultinomialNB(alpha = 0)

	scores = cross_val_score(clf, data_total, target, k)
	mean_score = [0.0, 0.0, 0.0]
	for sc in scores:
		for i in range(len(mean_score)):
			mean_score[i] += sc[i]
	for i in range(len(mean_score)):
		mean_score[i] /= k 
	print 'paragraph size = ' + str(data_size)
	print scores
	print '----------'
	print mean_score

print "more"
