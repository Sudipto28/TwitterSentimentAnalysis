import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.classify import ClassifierI
from statistics import mode

# Importing Naive Bayes classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

# To ignore FutureWarnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)  # Get vote for each classifier
            votes.append(v)

        return mode(votes)  # Returns the most occurring element from the list of votes

    def confidence(self, featureset):
        votes = []
        for c in self._classifiers:
            v = c.classify(featureset)
            votes.append(v)

        choice_votes = votes.count(mode(votes))     # Counts the total number of most occurring element in the list
        conf = choice_votes / len(votes)

        return conf


short_pos = open('positive.txt', 'r').read()
short_neg = open('negative.txt', 'r').read()

documents = []
all_words = []

#  j is adjective, r is adverb, and v is verb
# allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for r in short_pos.split('\n'):     # Since the data-set is separated by a new line (\n)
    documents.append((r, 'pos'))
    words = word_tokenize(r)
    pos = pos_tag(words)

    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):     # Since the data-set is separated by a new line (\n)
    documents.append((r, 'neg'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)

    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)
#
# for w in short_pos_words:
#     all_words.append(w.lower())
#
# for w in short_neg_words:
#     all_words.append(w.lower())

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()


all_words = nltk.FreqDist(all_words)    # Returns the key,value pair of the most common word to the least common word.

word_features = list(all_words.keys())[:5000]   # Returns the top 5000 words without the keys

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


# Find the features within the document that we are using
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)    # Returns True if one of the top 3000 words(from word_features) is in the document

    return features


featuresets = [(find_features(review), category) for (review, category) in documents]

save_featuresets = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Implementing Naive Bayes Classifier.
NB_classifier = nltk.NaiveBayesClassifier.train(training_set)
NB_accuracy = (nltk.classify.accuracy(NB_classifier, testing_set)) * 100
print('Naive Bayes Classifier Accuracy: ', NB_accuracy)

save_NB_classifier = open('pickled_algos/naivebayes.pickle', 'wb')
pickle.dump(NB_classifier, save_NB_classifier)
save_NB_classifier.close()

# Implementing scikit-learn #

# Multinomial Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)  # Train the model
MNB_accuracy = (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100
print('Multinomial Naive Bayes Classifier Accuracy: ', MNB_accuracy)

# Pickling
save_MNB_classifier = open('pickled_algos/MNB_classifier.pickle', 'wb')
pickle.dump(MNB_classifier, save_MNB_classifier)
save_MNB_classifier.close()

# Bernoulli Naive Bayes
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)  # Train the model
BNB_accuracy = (nltk.classify.accuracy(BNB_classifier, testing_set)) * 100
print('Bernoulli Naive Bayes Classifier Accuracy: ', BNB_accuracy)

# Pickling
save_BNB_classifier = open('pickled_algos/BNB_classifier.pickle', 'wb')
pickle.dump(BNB_classifier, save_BNB_classifier)
save_BNB_classifier.close()

# Logistic Regression Classifier
LR_classifier = SklearnClassifier(LogisticRegression())
LR_classifier.train(training_set)
LR_accuracy = (nltk.classify.accuracy(LR_classifier, testing_set)) * 100
print('Logistic Regression Classifier Accuracy: ', LR_accuracy)

# Pickling
save_LR_classifier = open('pickled_algos/LR_classifier.pickle', 'wb')
pickle.dump(LR_classifier, save_LR_classifier)
save_LR_classifier.close()

# Stochastic Gradient Descent (SGD) Classifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)
SGD_accuracy = (nltk.classify.accuracy(SGD_classifier, testing_set)) * 100
print('Stochastic Gradient Descent (SGD) Classifier Accuracy: ', SGD_accuracy)

# Pickling
save_SGD_classifier = open('pickled_algos/SGD_classifier.pickle', 'wb')
pickle.dump(SGD_classifier, save_SGD_classifier)
save_SGD_classifier.close()

# Linear SVC
LSVC_classifier = SklearnClassifier(LinearSVC())
LSVC_classifier.train(training_set)
LSVC_accuracy = (nltk.classify.accuracy(LSVC_classifier, testing_set)) * 100
print('LinearSVC Accuracy: ', LSVC_accuracy)

# Pickling
save_LSVC_classifier = open('pickled_algos/LSVC_classifier.pickle', 'wb')
pickle.dump(LSVC_classifier, save_LSVC_classifier)
save_LSVC_classifier.close()

# Nu SVC
NSVC_classifier = SklearnClassifier(NuSVC())
NSVC_classifier.train(training_set)
NSVC_accuracy = (nltk.classify.accuracy(NSVC_classifier, testing_set)) * 100
print('NuSVC Accuracy: ',  NSVC_accuracy)

# Pickling
save_NSVC_classifier = open('pickled_algos/NSVC_classifier.pickle', 'wb')
pickle.dump(NSVC_classifier, save_NSVC_classifier)
save_NSVC_classifier.close()

print('')
print('-----------------------------------------------------------------------------------------------------')
print('')

voted_classifier = VoteClassifier(NB_classifier, MNB_classifier, BNB_classifier, LR_classifier, SGD_classifier,
                                  LSVC_classifier, NSVC_classifier)
voted_classifier_accuracy = (nltk.classify.accuracy(voted_classifier, testing_set)) * 100
print('Voted Classifier Accuracy: ', voted_classifier_accuracy)
