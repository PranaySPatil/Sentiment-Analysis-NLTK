__author__ = 'Pranay'
import pickle
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

class BayesClassifier:
    def __init__(self):
        try:
            with open('naiveBayesClassifier.pickle', 'r+b') as f:
                self.classifier = pickle.load(f)
                print 'classifier loaded'
        except:
            self.classifier = None
            print 'classifier not found'

    def create_word_features(self, words):
        useful_words = [word for word in words if word not in
                        stopwords.words('english')]
        my_dict = dict([(word,True) for word in useful_words])
        return my_dict

    def train_classifier(self):
        neg_tweets = []
        try:
            with open('neg_tweets.pickle', 'r+b')as f:
                neg_tweets = pickle.load(f)
        except IOError:
            i = 1
            for string in twitter_samples.strings('negative_tweets.json'):
                print ('Negative Tweet - '+str(i))
                i+=1
                string = string.replace(':','').replace(';','')
                words = word_tokenize(string)
                neg_tweets.append((self.create_word_features(words), "negative"))
                with open('neg_tweets.pickle', 'w+b') as fw:
                    pickle.dump(neg_tweets, fw)
        pos_tweets = []
        try:
            with open('pos_tweets.pickle', 'r+b')as f:
                pos_tweets = pickle.load(f)
        except IOError:
            i = 1
            for string in twitter_samples.strings('positive_tweets.json'):
                print ('Positive Tweet - '+str(i))
                i+=1
                string = string.replace(':','').replace(';','')
                words = word_tokenize(string)
                pos_tweets.append((self.create_word_features(words), "positive"))
                with open('pos_tweets.pickle', 'w+b') as fw:
                    pickle.dump(pos_tweets, fw)
        train_set = neg_tweets[:4000] + pos_tweets[:4000]
        test_set =  neg_tweets[4000:] + pos_tweets[4000:]
        print 'Training Started...'
        self.classifier = NaiveBayesClassifier.train(train_set)
        with open('naiveBayesClassifier.pickle', 'w+b') as f:
            pickle.dump(self.classifier, f)
        self.accuracy = nltk.classify.util.accuracy(self.classifier, test_set)
        print 'Model trained'
        print 'Accurcy: ' + str(self.accuracy)

    def classify(self, stmnt):
        if not self.classifier:
            return 'Train the model first'
        words = word_tokenize(stmnt)
        words = self.create_word_features(words)
        return self.classifier.classify(words)

if __name__ == "__main__":
    bClassifier = BayesClassifier()
    # bClassifier.train_classifier()
    tweet1 = '''Jeremy Vine doesn't think the SNP Scottish takeover merits a mention'''
    tweet2 = '''Kind of like/support nick clegg hahahaha'''
    print(tweet1)
    print('Analysis Result :'+bClassifier.classify(tweet1))
    print(tweet2)
    print('Analysis Result :'+bClassifier.classify(tweet2))
