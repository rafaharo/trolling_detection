import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def train_basic(categories, comments):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
    #text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True)),
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
            vec += w2vmodel[word].reshape((1, size))
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
        self.badwords_ = badwords

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'allcaps', 'max_len',
            'mean_len', '@', '!', 'spaces', 'bad_ratio', 'n_bad',
            'capsratio'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        n_words = [len(c.split()) for c in documents]
        n_chars = [len(c) for c in documents]
        # number of uppercase words
        allcaps = [np.sum([w.isupper() for w in comment.split()])
               for comment in documents]
        # longest word
        max_word_len = [np.max([len(w) for w in c.split()]) for c in documents]
        # average word length
        mean_word_len = [np.mean([len(w) for w in c.split()])
                                            for c in documents]
        # number badwords:
        n_bad = [np.sum([c.lower().count(w) for w in self.badwords_])
                                                for c in documents]
        exclamation = [c.count("!") for c in documents]
        addressing = [c.count("@") for c in documents]
        spaces = [c.count(" ") for c in documents]

        allcaps_ratio = np.array(allcaps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, allcaps, max_word_len,
            mean_word_len, exclamation, addressing, spaces, bad_ratio, n_bad,
            allcaps_ratio]).T


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
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier

    text_clf = Pipeline([('vect', CustomTransformer(badwords)),
    #text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True)),
                      #('clf', LogisticRegression(tol=1e-8, penalty='l2', C=4))])
                      ('clf', AdaBoostClassifier(n_estimators=100))]) # DecisionTree by default
    text_clf = text_clf.fit(comments, categories)
    return text_clf


def train_assembling(categories, comments, badwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_selection import SelectPercentile, chi2

    select = SelectPercentile(score_func=chi2, percentile=16)
    tfidf = TfidfVectorizer(lowercase=True, min_df=3)
    countvect_word = TfidfVectorizer(ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
    custom = CustomTransformer(badwords)
    union = FeatureUnion([("custom", custom), ("tfidf", tfidf),("words", countvect_word)])
    #clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=4)
    classifier = Pipeline([('vect', union), ('select', select), ('clf', clf)])
    classifier = classifier.fit(comments, categories)
    return classifier


def train_assembling_average(categories, comments, badwords):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import AdaBoostClassifier

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3, n_iter=5, random_state=42))])

    custom = CustomTransformer(badwords)
    base_estimator = LogisticRegression(tol=1e-8, penalty='l2', C=4, solver='lbfgs')
    clf = Pipeline([('vect', custom),
                      ('clf', AdaBoostClassifier(n_estimators=100))])

    final_classifier = AverageClassifier([text_clf, clf])
    final_classifier = final_classifier.fit(comments, categories)
    return final_classifier