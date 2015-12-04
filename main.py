import json
import progressbar

feature_lst=[]
review_id_lst = []
# ------------each function in this part returns a feature list for all records---------
# input: a list l of json records
# output: consider to use dictionaty as output. Key will be the review id. 
def reviewLength(l):
	result = dict()

	return result

# input: a list l of json records
# output: consider to use dictionaty as output. Key will be the review id. 
def averageSentenceLength(l):
	result = dict()

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
# input:  a list l of json records
# output: a feature matrix object ready for training 
def preprocess(l):
	result = [[]]
	
	review_length = reviewLength(l)
	average_sentence_length = averageSentenceLength(l)
	#... other features

	# for each review record, get the result of each feature, and append to feature matrix
	for k in review_id_lst:
		record = []
		record.append(review_length(k))
		record.append(average_sentence_length(k))
		# ... append other features to the list

	result.append(record)
	return result

# ----------------training------------------
def main:
	raw_reviews = []

	rev_size = 18000
	# progress bar in terminal
	bar = progressbar.ProgressBar(maxval = len(rev_size) , \
        widgets=[progressbar.Bar('=','[',']'), ' ', progressbar.Percentage()])
	print('Opening json document for parsing')
	bar.start()
	i = 0
	with open('reviews_Baby.jsony.json') as f:
	    for line in f:
	    	if i >= 18000:
	    		break
	    	git.update(i + 1)
	    	i += 1
	       	raw_reviews.append(json.loads(line))
	bar.finish()
	print('Parsing finished')
    preprocess(raw_reviews)


if __name__ == '__main__':
    main()