__author__ = 'Pranay'
import pickle
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
        neg_reviews = []
        try:
            with open('neg_reviews.pickle', 'r+b')as f:
                neg_reviews = pickle.load(f)
        except IOError:
            for fileid in movie_reviews.fileids('neg'):
                words = movie_reviews.words(fileid)
                neg_reviews.append((self.create_word_features(words), "negative"))
                with open('neg_reviews.pickle', 'w+b') as fw:
                    pickle.dump(neg_reviews, fw)
        pos_reviews = []
        try:
            with open('pos_reviews.pickle', 'r+b')as f:
                pos_reviews = pickle.load(f)
        except IOError:
            for fileid in movie_reviews.fileids('pos'):
                words = movie_reviews.words(fileid)
                pos_reviews.append((self.create_word_features(words), "positive"))
                with open('pos_reviews.pickle', 'w+b') as fw:
                    pickle.dump(pos_reviews, fw)
        train_set = neg_reviews[:750] + pos_reviews[:750]
        test_set =  neg_reviews[750:] + pos_reviews[750:]
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
    review_spirit = '''
    Spirited Away' is the first Miyazaki I have seen, but from this stupendous film I can tell he is a master storyteller. A hallmark of a good storyteller is making the audience empathise or pull them into the shoes of the central character. Miyazaki does this brilliantly in 'Spirited Away'. During the first fifteen minutes we have no idea what is going on. Neither does the main character Chihiro. We discover the world as Chihiro does and it's truly amazing to watch. But Miyazaki doesn't seem to treat this world as something amazing. The world is filmed just like our workaday world would. The inhabitants of the world go about their daily business as usual as full with apathy as us normal folks. Places and buildings are not greeted by towering establishing shots and majestic music. The fact that this place is amazing doesn't seem to concern Miyazaki.
    What do however, are the characters. Miyazaki lingers upon the characters as if they were actors. He infixes his animated actors with such subtleties that I have never seen, even from animation giants Pixar. Twenty minutes into this film and I completely forgot these were animated characters; I started to care for them like they were living and breathing. Miyazaki treats the modest achievements of Chihiro with unashamed bombast. The uplifting scene where she cleanses the River God is accompanied by stirring music and is as exciting as watching gladiatorial combatants fight. Of course, by giving the audience developed characters to care about, the action and conflicts will always be more exciting, terrifying and uplifting than normal, generic action scenes.
    '''
    print(review_spirit)
    print('Analysis Result :'+ bClassifier.classify(review_spirit))

    review_santa = '''
    It would be impossible to sum up all the stuff that sucks about this film, so I'll break it down into what I remember most strongly: a man in an ingeniously fake-looking polar bear costume (funnier than the "bear" from Hercules in New York); an extra with the most unnatural laugh you're ever likely to hear; an ex-dope addict martian with tics; kid actors who make sure every syllable of their lines are slowly and caaarreee-fulll-yyy prrooo-noun-ceeed; a newspaper headline stating that Santa's been "kidnaped", and a giant robot. Yes, you read that right. A giant robot.
    The worst acting job in here must be when Mother Claus and her elves have been "frozen" by the "Martians'" weapons. Could they be *more* trembling? I know this was the sixties and everyone was doped up, but still.
    '''
    print(review_santa)
    print('Analysis Result :'+ bClassifier.classify(review_santa))
