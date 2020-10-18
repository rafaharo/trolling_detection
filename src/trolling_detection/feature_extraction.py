import spacy
import numpy as np
from sklearn.base import BaseEstimator

nlp = spacy.load("en_core_web_sm")

def corpus_stats(collection):
    import nltk
    import pprint
    words = tokenize_collection(collection, lowercase=True, stopwords='english', min_length=3)
    text = nltk.Text(word.lower() for word in words)
    print("Number of Words: " + str(len(text)))
    print("Number of unique words: " + str(len(set(text))))
    dist = nltk.FreqDist(text)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dist.most_common(20))


def tf_idf_stats(collection, num=20):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer='word', min_df=3, stop_words='english')
    matrix = tfidf.fit_transform(collection)
    dense = matrix.todense()
    features_names = tfidf.get_feature_names()
    print("\nNumber of Features: " + str(len(features_names)))
    for index, row in enumerate(dense[0:num]):
        print("\nScores for comment: " + str(index))
        comment = row.tolist()[0]
        scores = [pair for pair in zip(range(0, len(comment)), comment) if pair[1] > 0]
        sorted_scores = sorted(scores, key=lambda t: t[1] * -1)
        for phrase, score in [(features_names[word_id], score) for (word_id, score) in sorted_scores][:num]:
            print('{0: <20} {1}'.format(phrase, score))
        print("\n")


def similar_words(collection, word, num=10):
    import nltk
    words = tokenize_collection(collection, stopwords='english')
    text = nltk.Text(word.lower() for word in words)
    text.similar(word, num)


def language_model(collection):
    from nltk import ConditionalProbDist
    from nltk import ConditionalFreqDist
    from nltk import bigrams
    from nltk import MLEProbDist
    words = tokenize_collection(collection)
    freq_model = ConditionalFreqDist(bigrams(words))
    prob_model = ConditionalProbDist(freq_model, MLEProbDist)
    return prob_model


def word2vec_model(documents, n_dim=1000):
    from gensim.models import Word2Vec
    model = Word2Vec(documents, size=n_dim, window=8, min_count=3, workers=4)
    return model


def tokenize_collection(collection, lowercase=True, stopwords=True, min_length=3):
    documents = [tokenize_document(document, lowercase=lowercase, stopwords=stopwords, min_length=min_length)
                 for document in collection]
    words = [token for document in documents for token in document]
    return words


def tokenize_document(document, lowercase=True, stopwords=None, min_length=3):
    import nltk
    if not document or len(document) == 0:
        raise ValueError("Can't tokenize null or empty texts")

    if lowercase:
        document = document.lower()

    tokens = nltk.wordpunct_tokenize(document)

    if stopwords and isinstance(stopwords, str):
        stops = set(nltk.corpus.stopwords.words(stopwords))
    elif stopwords and isinstance(stopwords, list):
        stops = set(stopwords)
    else:
        stops = set()

    result = [token for token in tokens if token not in stops and len(token) >= min_length]
    return result


def replace_badwords(comment, badwords):
    comment = comment.lower()
    candidates = [insult for insult in badwords if insult in comment]
    candidates_sorted = sorted(candidates, key=len)
    for candidate in candidates_sorted:
        comment = comment.replace(candidate, " fakeinsult ")
    return comment


class CustomTransformer(BaseEstimator):
    def __init__(self):
        from spacy.matcher import Matcher
        self.__matcher = Matcher(nlp.vocab)

        pattern1 = [{"LEMMA": "-PRON-", "LOWER": {"IN": ["you", "your"]}},
                    {"LEMMA": {"IN": ["be", "sound"]}}, {"OP": "*", "LENGTH": {"<=": 10}},
                    {"LOWER": "fakeinsult"}]
        self.__matcher.add("insult1", None, pattern1)

        pattern2 = [{"LEMMA": "-PRON-", "LOWER": {"IN": ["you", "your"]}},
                    {"OP": "*", "LENGTH": {"<=": 4}},
                    {"LOWER": "fakeinsult"}]
        self.__matcher.add("insult2", None, pattern2)

    def get_feature_names(self):
        return np.array(['n_words', 'n_chars', 'n_dwords', 'you_re', '!', 'allcaps', '@', 'bad_ratio', 'n_bad',
                         'capsratio', 'dicratio' 'sent'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        import enchant
        import sentlex
        from feature_extraction import tokenize_document

        d = enchant.Dict("en_US")
        swn = sentlex.SWN3Lexicon()
        tokenized_documents = [tokenize_document(document) for document in documents]
        n_words = []
        n_chars = []
        # number of uppercase words
        all_caps = []
        n_bad = []
        exclamation = []
        addressing = []

        n_dwords = [sum(1 for word in document if d.check(word)) for document in tokenized_documents]

        sent_pos = []
        sent_neg = []
        n_you_re = []
        for comment in documents:
            n_words.append(len(comment.split()))
            n_chars.append(len(comment))
            all_caps.append(np.sum([w.isupper() for w in comment.split()]))
            n_bad.append(comment.count('fakeinsult'))
            exclamation.append(comment.count("!"))
            addressing.append(comment.count("@"))
            doc = nlp(comment)
            count = 0.
            pos_sum = 0.
            neg_sum = 0.
            for token in doc:
                if token.text == 'fakeinsult':
                    pos_sum += 0.
                    neg_sum += 1.
                    count += 1.
                    continue
                if token.pos_.startswith('RB'):
                    sentiment = swn.getadverb(token.text)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                elif token.pos_.startswith('NN'):
                    sentiment = swn.getnoun(token.text)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                if token.pos_.startswith('JJ'):
                    sentiment = swn.getadjective(token.text)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
                if token.pos_.startswith('VB'):
                    sentiment = swn.getverb(token.text)
                    pos_sum += sentiment[0]
                    neg_sum += sentiment[1]
                    count += 1.
            if count != 0:
                pos_sum /= count
                neg_sum /= count
            sent_neg.append(neg_sum)
            sent_pos.append(pos_sum)
            matches = self.__matcher(doc)
            n_you_re.append(len(matches))

        allcaps_ratio = np.array(all_caps) / np.array(n_words, dtype=np.float)
        bad_ratio = np.array(n_bad) / np.array(n_words, dtype=np.float)
        dic_ratio = np.array(n_dwords) / np.array(n_words, dtype=np.float)

        return np.array([n_words, n_chars, n_dwords, n_you_re, exclamation, all_caps,
                         addressing, bad_ratio, n_bad, allcaps_ratio, dic_ratio,
                         sent_pos]).T

    def get_params(self, deep=True):
        if not deep:
            return super(CustomTransformer, self).get_params(deep=False)
        else:
            out = super(CustomTransformer, self).get_params(deep=False)
            out.update(self.__matcher.copy())
            return out