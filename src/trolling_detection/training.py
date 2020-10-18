import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

def train_tfidf(comments, categories, class_weight=None):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42,
                                               class_weight=class_weight))])
    text_clf = text_clf.fit(comments, categories)
    return text_clf


class NaiveClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, badwords=[], fake=True):
        self.badwords = badwords
        self.fake = fake

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):
        predictions = []
        for x in X:
            if not self.fake:
                if any(badword in x.lower() for badword in self.badwords):
                    predictions.append(1)
                else:
                    predictions.append(0)
            else:
                if "fakeinsult" in x:
                    predictions.append(1)
                else:
                    predictions.append(0)
        return np.array(predictions)

def build_word_vector(w2vmodel, text, size):
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
    vecs = np.concatenate([build_word_vector(model, z, n_dim) for z in collection])
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


def train_custom(comments, categories):
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from feature_extraction import CustomTransformer

    text_clf = Pipeline([('vect', CustomTransformer()),
                         ('clf', LinearSVC(random_state=42, dual=False))])
    text_clf = text_clf.fit(comments, categories)
    return text_clf


def train_feature_union(comments, categories):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_selection import SelectPercentile, chi2
    from feature_extraction import CustomTransformer

    select = SelectPercentile(score_func=chi2, percentile=70)
    countvect_word = TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", binary=False, min_df=3)
    custom = CustomTransformer()
    union = FeatureUnion([("custom", custom), ("words", countvect_word)])
    clf = LogisticRegression(tol=1e-8, penalty='l2', C=4, max_iter=10000)
    classifier = Pipeline([('vect', union), ('select', select), ('clf', clf)])
    classifier = classifier.fit(comments, categories)
    return classifier


def train_assembling_voting(comments, categories):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import VotingClassifier
    from feature_extraction import CustomTransformer

    text_clf = Pipeline([('vect', TfidfVectorizer(lowercase=True, ngram_range=(1, 3), analyzer="word", min_df=3)),
                         ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-3,
                                               random_state=42))])

    custom = CustomTransformer()
    clf = Pipeline([('vect', custom),
                    ('clf', SGDClassifier(loss='log', penalty='l2',
                                          alpha=1e-3, random_state=42))])

    final_classifier = VotingClassifier(estimators=[('text', text_clf), ('custom', clf)],
                                        voting='soft', weights=[3, 1])
    final_classifier = final_classifier.fit(comments, categories)
    return final_classifier
