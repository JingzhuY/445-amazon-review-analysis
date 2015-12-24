import json
import re
import progressbar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
import random
from sklearn import svm
import numpy
import csv
from sklearn import preprocessing
import numpy as np
#from collections import Counter

feature_lst=[]
review_id_lst = []
# ------------each function in this part returns a feature list for all records---------
# input: a list l of json records
# output: returns the word count of each review
def reviewLength(l):
	result = []
	print('Review Length function initiating')
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	i = 0
	bar.start()
	for review in l:
		bar.update(i + 1)
		i+=1
		result.append( len(review['reviewText'].split()) )
	bar.finish()
	return result

# input: a list l of json records
# output: returns the average length of a sentence in a review 
def averageSentenceLength(l):
	result = []
	print ('Counting average sentence length')
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	for review in l:
		review_text = review['reviewText']
		sentences = [review_text.strip() for review_text in re.split('[\.\?!]' , review_text) if review_text]
		if len(sentences)==0:
			print "text: "+review_text
		word_sum = 0
		bar.update(i + 1)
		i+=1
		for w in sentences:
			word_sum += len( w.split() )
		result.append( float(word_sum) / len(sentences) )
	bar.finish()
	return result

# input: a list l of json records
# output: returns the count of exclamation and question marks in a review
def punctuationCount(l):
	print 'counting punctuation ratio'
	result = []
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	puctuation=["!","?"]
	for review in l:
		bar.update(i + 1)
		i+=1
		c = sum(1 for c in review['reviewText'] if c in puctuation)
		#result.append(c['!']+c['?'])
		result.append(c)
	bar.finish()
	return result

# input: a list l of json records
# output: returns the count of words in all caps
def capWordCount(l):
	result = []
	print 'counting words in all caps'
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	for review in l:
		bar.update(i + 1)
		i+=1
		c = sum(1 for c in review['reviewText'].split() if c==c.upper())
		result.append(c)
	bar.finish()
	return result

# input: a list l of json records
# output: the count of words in title(summary)
def titleWordsCount(l):
	result = []
	print 'counting title words'
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	for review in l:
		i+=1
		result.append( len(review['summary'].split()) )
	bar.finish()
	return result

# input: a list
# output: review rating
def getRating(l):
	result = []
	print 'getting review ratings'
	bar = progressbar.ProgressBar(maxval = len(l) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	bar.start()
	i = 0
	for review in l:
		i += 1
		result.append(review['overall'])
	bar.finish()
	return result

# input:  a list l of string
# output: a matrix where the (i,j) component is how many times 
#         the j-th word appear in the i-th document
def tf(l):
    result = []
    vectorizer = CountVectorizer(min_df=1)
    result = vectorizer.fit_transform(l).toarray()
    return result

# input:  a list l of string
# output: a matrix where the (i,j) component is the tf-idf value of the j-th word in the i-th document
def tfidf(l):
    result = []
    vectorizer = TfidfVectorizer(min_df = 1)
    result = vectorizer.fit_transform(l).toarray()
    return result

def word2VecScore(l):
	result = []
	reviews = []

	for product in l:
		reviews.append(product['reviewText'])

	sentences = []

	for review in reviews:
		sentences.append(review.split())

	model = Word2Vec(sentences, size = 100, window = 5, min_count = 1, workers = 4)

	for review in reviews:
		vector_average = 0
		words = review.split()
		for word in words:
			vector_average = vector_average + model[word]
		vector_average = vector_average / len(words)
		result.append(vector_average)

	return result

# input: a list of training text and training labels
# output: a list of probabilities that a review is helpful
def word2VecProb(l, helpfulness):
	print 'calculating word2vec prob'
	# separate training(40%), validation(30%) and testing(30%) sets
	reviews = []
	labels = []

	for product in l:
		reviews.append(product['reviewText'])

	length = len(reviews)
	valid_len = length / 10 * 3
	test_len = length / 10 * 3
	train_len = length - valid_len - test_len

	valid_text = reviews[:valid_len]
	test_text = reviews[valid_len:(valid_len+test_len)]
	train_text = reviews[(valid_len+test_len):]

	sentences = []
	for review in train_text:
		sentences.append(review.split())
	for review in test_text:
		sentences.append(review.split())
	for review in valid_text:
		sentences.append(review.split())


	# get train label from "helpfulness"
	for h in helpfulness:
		hrate = float(h[0])/h[1]
		#label.append(hrate)
		if hrate > 0.6: # 0.6: 53% positive
			labels.append(1) #possitive
		else:
			labels.append(0) #negative

	train_labels = labels[(valid_len+test_len):]
	valid_labels = labels[:valid_len]


	# trains word2vec
	model = Word2Vec(sentences, size = 100, window = 5, min_count = 1, workers = 4)

	# creates features for training data
	train_features = []
	for review in train_text:
		vector_average = 0
		words = review.split()
		for word in words:
			vector_average = vector_average + model[word]
		vector_average = vector_average / len(words)
		train_features.append(vector_average)

	# creates features for test data
	test_features = []
	for review in test_text:
		vector_average = 0
		words = review.split()
		for word in words:
			vector_average = vector_average + model[word]
		vector_average = vector_average / len(words)
		test_features.append(vector_average)

	# creates features for test data
	valid_features = []
	for review in valid_text:
		vector_average = 0
		words = review.split()
		for word in words:
			vector_average = vector_average + model[word]
		vector_average = vector_average / len(words)
		valid_features.append(vector_average)

	# trains logistic regression model
	lr=LogisticRegression(C=1, penalty='l2')
	lr.fit(train_features, train_labels)

	# returns predicted probabilities of the test reviews

	valid_prob = lr.predict_proba(valid_features) 
	test_prob = lr.predict_proba(test_features) 
	train_prob = lr.predict_proba(train_features)

	return numpy.concatenate((valid_prob[:,1], test_prob[:,1], train_prob[:,1]), axis = 0)


#------------preprocess prepare the feature matrix for training---------------
# input:  a list l of json records, and a list of helpfulness labels
# output: train_x, valid_x, test_x, train_t, valid_t, test_t
def preprocess(l, helpfulness):
	result = []
	label = []
	# result list
	review_length = reviewLength(l)
	average_sentence_length = averageSentenceLength(l)
	punctuation_count = punctuationCount(l)
	cap_word_count = capWordCount(l)
	word2vec_score = word2VecScore(l)
	title_word_count = titleWordsCount(l)
	rating = getRating(l)
	#... other features

	# for each review record, get the result of each feature, and append to feature matrix
	for k in range(0,len(l)):
		record = []
		# ... append other features to the list
		record.append(review_length[k])
		record.append(average_sentence_length[k])
		record.append(punctuation_count[k])
		record.append(cap_word_count[k])
		record.append(title_word_count[k])
		record.append(rating[k])
		record.extend(word2vec_score[k])
		result.append(record)

	# scaling
	result = np.array(result)
	result = preprocessing.scale(result)
	#get train label from "helpfulness"
	for h in helpfulness:
		if h[0] > h[1] * .6: #if more than half rated helpful
			label.append(1) #positive
		else:
			label.append(-1) #negative
		hrate = float(h[0])/h[1]
		#label.append(hrate)


	# seperate training(40%), validation(30%) and testing(30%) sets
	length = len(result)
	valid_len = length / 10 * 3
	test_len = length / 10 * 3
	train_len = length - valid_len - test_len

	valid_x = result[:valid_len]
	test_x = result[valid_len:(valid_len+test_len)]
	train_x = result[(valid_len+test_len):]

	valid_t = label[:valid_len]
	test_t = label[valid_len:(valid_len+test_len)]
	train_t = label[(valid_len+test_len):]
	return (train_x, valid_x, test_x, train_t, valid_t, test_t)

def logisticRegression(train_X, train_t, val_X, val_t):
	C=[0.1, 1, 10, 100, 1000]
	for c in C:
		lr=LogisticRegression(C=c, penalty='l2')
		train_X = numpy.array(train_X)
		val_X = numpy.array(val_X)
		pred_val_t=lr.fit(train_X,train_t).predict(val_X)

		# calculat accuracy
		mismatch=0
		for r,p in zip(val_t, pred_val_t):
			if r!=p:
				mismatch+=1
		accuracy=1-float(mismatch)/len(val_t)
		print "C="+str(c)+", accuracy="+ str(accuracy)

def SVM(train_X, train_t, val_X, val_t):
    C=[0.1, 1, 10, 100]
    for c in C:
        clf = svm.SVC(C=c, kernel='rbf') # change kernels
        #train_X=csr_matrix(train_X)
        #val_X=csr_matrix(val_X)
        train_X = numpy.array(train_X)
        val_X = numpy.array(val_X)
        pred_val_t=clf.fit(train_X,train_t).predict(val_X)


        mismatch=0
        for r,p in zip(val_t, pred_val_t):
            if r!=p:
                mismatch+=1
        accuracy=1-float(mismatch)/len(val_t)
        print "C="+str(c)+", accuracy="+ str(accuracy)
# ----------------training------------------
def main():
	raw_reviews = []
	helpfulness = []
	rev_size = 50000
	# progress bar in terminal
	bar = progressbar.ProgressBar(maxval = rev_size , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	print('Opening json document for parsing')
	bar.start()
	i = 0
	i_pos = 0
	i_neg = 0
	#with open('reviews_Baby.jsony.json') as f:
	with open('reviews_Movies_and_TV.json') as f:
	    for line in f:
	#    	if i >= rev_size:
	 #   		break
	  #  	bar.update(i + 1)
	    	if i_pos + i_neg >= rev_size:
	    		break
	    	bar.update(i_pos + i_neg)
	    	
	    	line = json.loads(line)
	    	# skip the lines with empty review
	    	# skip the lines with ratings of helpfulness less than a threshold (5 for now)
	    	
	    	helpfulThreshold = 10
	    	if len(line['reviewText']) != 0 and line['helpful'][1] >= helpfulThreshold:
	    		i += 1
	    		if line['helpful'][0] > line['helpful'][1] * .6: #if more than 60% helpful
	    			if i_pos < rev_size/2:
	    				i_pos += 1
	    				raw_reviews.append(line)
	    				helpfulness.append(line['helpful'])
	    		else:
	    			if i_neg < rev_size/2:
		    			i_neg += 1
		    			raw_reviews.append(line)
	    				helpfulness.append(line['helpful'])
	    		#raw_reviews.append(line)
	    		#helpfulness.append(line['helpful'])
	    		#i += 1



					
	bar.finish()
	print('Parsing finished')
	# get the feature matrix and labels
	(train_x, valid_x, test_x, train_t, valid_t, test_t)= preprocess(raw_reviews, helpfulness)
	# for testing
	
	print '-------logisticRegression------'
	logisticRegression(train_x, train_t, valid_x, valid_t)
	print '------SVM------'
	SVM(train_x, train_t, valid_x, valid_t)
	'''
	UPDATE:
	added two features: title word count, review rating. Two features removed: cap words count, punctuation count
	label is possitive if helpfulness rate (rated helpful/ total) greater than 0.6
	
	adjuct the labeling strategy (how to decide whether or not to drop the record, and whether it's helpful or not helpful)

	'''
	

if __name__ == '__main__':
    main()
