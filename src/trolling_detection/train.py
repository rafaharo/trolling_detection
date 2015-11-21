import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def train_basic(categories, comments):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])
    text_clf = text_clf.fit(comments, categories)
    return text_clf


def naive_classifier(comment, badwords, fake=False):
    if not fake:
        if any(badword in comment.lower() for badword in badwords):
            return 1
        else:
            return 0
    else:
        if "fakeinsult" in comment:
            return 1
        else:
            return 0

def buildWordVector(w2vmodel, text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            sorted_vec = np.sort(w2vmodel[word])
            vec += sorted_vec.reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def w2vectorize(collection, model, n_dim):
    from sklearn.preprocessing import scale
    vecs = np.concatenate([buildWordVector(model, z, n_dim) for z in collection])
    vecs = scale(vecs)
    return vecs

def train_word2vec(categories, comments, n_dim):
    from feature_extraction import tokenize_document
    from feature_extraction import word2vec_model
    from sklearn.linear_model import SGDClassifier
    documents = [tokenize_document(document) for document in comments]
    model = word2vec_model(documents, n_dim)
    train_vecs = w2vectorize(documents, model, n_dim)
    classifier = SGDClassifier(loss='log', penalty='l1')
    classifier.fit(train_vecs, categories)

    return model, classifier


class CustomTransformer(BaseEstimator):
    def __init__(self, badwords):
        if badwords:
            self.__badwords = badwords
        else:
            from main import load_bad_words_static
            self.__badwords = load_bad_words_static()
        self.__you_re = "(you|you are|you're|you sound)[\w\s]{1,40}\sfakeinsult"
        self.__you = "you're|you are|you sound"

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'n_dwords', 'you_re', 'allcaps', '@', '!', 'bad_ratio', 'n_bad',
            'capsratio', 'dicratio' 'sent'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        import enchant
        import re
        import sentlex
        from pattern.en import tag as tagger

        d = enchant.Dict("en_US")
        SWN = sentlex.SWN3Lexicon()
        from feature_extraction import tokenize_document
        tokenized_documents = [tokenize_document(document) for document in documents]
        n_words = [len(c.split()) for c in documents]
        #n_words = [len(document) for document in tokenized_documents]
        n_chars = [len(c) for c in documents]
        n_dwords = [sum(1 for word in document if d.check(word)) for document in tokenized_documents]

        sent_pos = []
        sent_neg = []
        for comment in documents:
            count = 0.
            pos_sum = 0.
            neg_sum = 0.
            for word, tag in tagger(comment.lower()):
                if word == 'fakeinsult':
                    pos_sum += 0.
                    neg_sum += 1.
                    count += 1.
                    continue
                if tag.startswith('RB'):
                    sentiment = SWN.getadverb(word)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                elif tag.startswith('NN'):
                    sentiment = SWN.getnoun(word)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                if tag.startswith('JJ'):
                    sentiment = SWN.getadjective(word)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                if tag.startswith('VB'):
                    sentiment = SWN.getverb(word)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
            if count != 0:
                pos_sum /= count
                neg_sum /= count
            sent_neg.append(neg_sum)
            sent_pos.append(pos_sum)

        n_you_re = [len(re.findall(self.__you_re, document)) for document in documents]
        n_you = [len(re.findall(self.__you, document)) for document in documents]

        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        #max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        #mean_word_len = [np.mean([len(w) for w in c.split()])
        #                                    for c in documents]
        # number badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.__badwords]) for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)
        dic_ratio = np.array(n_dwords) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, n_dwords, n_you_re, n_you, exclamation, allcaps,
                         addressing, bad_ratio, n_bad, allcaps_ratio, dic_ratio,
                         sent_pos]).T

    def get_params(self, deep=True):
        if not deep:
            return super(CustomTransformer, self).get_params(deep=False)
        else:
            out = super(CustomTransformer, self).get_params(deep=False)
            out.update(self.__badwords.copy())
            out.update(self.__you_re.copy())
            out.update(self.__you.copy())
            return out

class AverageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return np.mean(self.predictions_, axis=0)

def train_custom(categories, comments, badwords):
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import GradientBoostingRegressor


    text_clf = Pipeline([('vect', CustomTransformer(badwords)),
                      ('clf', LinearSVC(random_state=42))])
    text_clf = text_clf.fit(comments, categories)
    return text_clf


def train_assembling(categories, comments, badwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_selection import SelectPercentile, chi2

    select = SelectPercentile(score_func=chi2, percentile=70)
    countvect_word = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
    custom = CustomTransformer(badwords)
    union = FeatureUnion([("custom", custom),("words", countvect_word)])
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
    classifier = Pipeline([('vect', union), ('select', select), ('clf', clf)])
    classifier = classifier.fit(comments, categories)
    return classifier


def train_assembling_average(categories, comments, badwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import VotingClassifier

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    custom = CustomTransformer(badwords)
    clf = Pipeline([('vect', custom),
                    ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    final_classifier = VotingClassifier(estimators=[('text', text_clf), ('custom', clf)],
                                        voting='soft', weights=[3,1])
    final_classifier = final_classifier.fit(comments, categories)
    return final_classifier
