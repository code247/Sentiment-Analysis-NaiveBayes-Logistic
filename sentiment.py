import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
import itertools
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
	print "python sentiment.py <path_to_data> <0|1>"
	print "0 = NLP, 1 = Doc2Vec"
	exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
	train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

	if method == 0:
	    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
	    nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
	if method == 1:
	    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
	    nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
	print "Naive Bayes"
	print "-----------"
	evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
	print ""
	print "Logistic Regression"
	print "-------------------"
	evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
	"""
	Loads the train and test set into four different lists.
	"""
	train_pos = []
	train_neg = []
	test_pos = []
	test_neg = []
	with open(path_to_dir+"train-pos.txt", "r") as f:
		for i,line in enumerate(f):
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			train_pos.append(words)
	with open(path_to_dir+"train-neg.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			train_neg.append(words)
	with open(path_to_dir+"test-pos.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			test_pos.append(words)
	with open(path_to_dir+"test-neg.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			test_neg.append(words)

	return train_pos, train_neg, test_pos, test_neg

def clean(dataset):
	distinct_words = []
	filtered_words = []
	stopwords = set(nltk.corpus.stopwords.words('english'))
	for text in dataset:
		distinct_words.append(list(set(text)))
	for text in distinct_words:
		filtered_words.append(list(set(text) - stopwords))
	flat_words = list(itertools.chain(*filtered_words))
	return flat_words

def create_vector(group, features):
	answer = []
	for item in group:
		vector = []
		for w in features:
			if w in item:
				vector.append(1)
			else:
				vector.append(0)
		answer.append(vector)
	return answer

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# English stopwords from nltk
	# stopwords = set(nltk.corpus.stopwords.words('english'))

	# Determine a list of words that will be used as features. 
	# This list should have the following properties:
	#   (1) Contains no stop words
	#   (2) Is in at least 1% of the positive texts or 1% of the negative texts
	#   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
	# YOUR CODE HERE

	ltrainpos = len(train_pos)
	ltrainneg = len(train_neg)
	ltestpos = len(test_pos)
	ltestneg = len(test_neg)

	clean_train_pos = clean(train_pos)
	clean_train_neg = clean(train_neg)
	clean_test_pos = clean(test_pos)
	clean_test_neg = clean(test_neg)

	c_train_pos = Counter(clean_train_pos)
	c_train_neg = Counter(clean_train_neg)
	c_test_pos = Counter(clean_test_pos)
	c_test_neg = Counter(clean_test_neg)

	train_pos_feat = filter(lambda (x, count) : (count >= int(0.01 * ltrainpos) and count >= 2 * c_train_neg[x]), c_train_pos.iteritems())
	train_neg_feat = filter(lambda (x, count) : (count >= int(0.01 * ltrainneg) and count >= 2 * c_train_pos[x]), c_train_neg.iteritems())
	
	temp_feat = train_pos_feat + train_neg_feat

	train_feat = map(lambda (x, count): x, temp_feat)

	# Using the above words as features, construct binary vectors for each text in the training and test set.
	# These should be python lists containing 0 and 1 integers.
	# YOUR CODE HERE

	train_pos_vec = create_vector(train_pos, train_feat)
	train_neg_vec = create_vector(train_neg, train_feat)
	test_pos_vec = create_vector(test_pos, train_feat)
	test_neg_vec = create_vector(test_neg, train_feat)

	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

def labelize_words(dataset, label_type):
	labelize = []
	for i, v in enumerate(dataset):
		label = '%s_%s'%(label_type,i)
		labelize.append(LabeledSentence(v, [label]))
	return labelize

def extract_features(model, labeled_data, label_type):
	vector = []
	for i in range(len(labeled_data)):
		vector.append(model.docvecs[label_type + str(i)])
	return vector

def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# Doc2Vec requires LabeledSentence objects as input.
	# Turn the datasets from lists of words to lists of LabeledSentence objects.
	# YOUR CODE HERE

	labeled_train_pos = labelize_words(train_pos, "TRAIN_POS")
	labeled_train_neg = labelize_words(train_neg, "TRAIN_NEG")
	labeled_test_pos = labelize_words(test_pos, "TEST_POS")
	labeled_test_neg = labelize_words(test_neg, "TEST_NEG")

	# Initialize model
	model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
	sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
	model.build_vocab(sentences)

	# Train the model
	# This may take a bit to run 
	for i in range(5):
		print "Training iteration %d" % (i)
		random.shuffle(sentences)
		model.train(sentences)

	# Use the docvecs function to extract the feature vectors for the training and test data
	# YOUR CODE HERE

	train_pos_vec = extract_features(model, labeled_train_pos, "TRAIN_POS_")
	train_neg_vec = extract_features(model, labeled_train_neg, "TRAIN_NEG_")
	test_pos_vec = extract_features(model, labeled_test_pos, "TEST_POS_")
	test_neg_vec = extract_features(model, labeled_test_neg, "TEST_NEG_")

	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
	"""
	Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
	# For BernoulliNB, use alpha=1.0 and binarize=None
	# For LogisticRegression, pass no parameters
	# YOUR CODE HERE

	X = train_pos_vec + train_neg_vec

	nb_model = BernoulliNB(alpha=1.0, binarize=None)
	nb_model.fit(X, Y)

	lr_model = LogisticRegression()
	lr_model.fit(X, Y)

	return nb_model, lr_model

def build_models_DOC(train_pos_vec, train_neg_vec):
	"""
	Returns a GaussianNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
	# For LogisticRegression, pass no parameters
	# YOUR CODE HERE

	X = train_pos_vec + train_neg_vec

	nb_model = GaussianNB()
	nb_model.fit(X, Y)

	lr_model = LogisticRegression()
	lr_model.fit(X, Y)

	return nb_model, lr_model

def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
	"""
	Prints the confusion matrix and accuracy of the model.
	"""
	# Use the predict function and calculate the true/false positives and true/false negative.
	# YOUR CODE HERE

	test = test_pos_vec + test_neg_vec

	prediction = model.predict(test)
	Y = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)

	confusion = confusion_matrix(Y, prediction)

	tp = confusion[1][1]

	fn = confusion[1][0]

	fp = confusion[0][1]

	tn = confusion[0][0]

	accuracy = float(tp + tn)/float(tp + fn + fp + tn)

	if print_confusion:
		print "predicted:\tpos\tneg"
		print "actual:"
		print "pos\t\t%d\t%d" % (tp, fn)
		print "neg\t\t%d\t%d" % (fp, tn)
	print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
