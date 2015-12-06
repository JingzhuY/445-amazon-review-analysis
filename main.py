import json
import re
import progressbar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec
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

# input: a list of training text and training labels
# output: a list of probabilities that a review is helpful
def probabilityWord2Vec(train_text, train_labels, test_text):
	sentences = []
	for review in train_text:
		sentences.append(review.split())
	for review in test_text:
		sentences.append(review.split())

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

	# trains logistic regression model
	model.fit(train_features, train_labels)

	# returns predicted probabilities of the test reviews
	return model.predict_proba(test_features)


#------------preprocess prepare the feature matrix for training---------------
# input:  a list l of json records, and a list of helpfulness labels
# output: a feature matrix object ready for training, and a list "train_t" of labels
def preprocess(l, helpfulness):
	result = []
	
	# vector of review lengths
	review_length = reviewLength(l)
	average_sentence_length = averageSentenceLength(l)
	punctuation_count = punctuationCount(l)
	cap_word_count = capWordCount(l)
	#... other features

	# for each review record, get the result of each feature, and append to feature matrix
	for k in range(0,len(l)):
		record = []
		record.append(review_length[k])
		record.append(average_sentence_length[k])
		record.append(punctuation_count[k])
		record.append(cap_word_count[k])
		# ... append other features to the list
		
		result.append(record)

	# get train label from "helpfulness"
	train_t = helpfulness #some function here
	return (result, train_t)

# ----------------training------------------
def main():
	raw_reviews = []
	helpfulness = []
	rev_size = 18000
	# progress bar in terminal
	bar = progressbar.ProgressBar(maxval = rev_size , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	print('Opening json document for parsing')
	bar.start()
	i = 0
	#with open('reviews_Baby.jsony.json') as f:
	with open('reviews_Sports_and_Outdoors.json') as f:
	    for line in f:
	    	if i >= rev_size:
	    		break
	    	bar.update(i + 1)
	    	i += 1
	    	line = json.loads(line)
	    	# skip the line whose review is empty
	    	if len(line['reviewText']) != 0:
	       		raw_reviews.append(line)
	       		helpfulness.append(line['helpful'])
	bar.finish()
	print('Parsing finished')
	# get the feature matrix and labels
	(feature_matrix, train_t)= preprocess(raw_reviews, helpfulness)
	print len(feature_matrix)
	print feature_matrix[:50]
	print train_t[:50]

if __name__ == '__main__':
    main()