def extract_top_bigrams_collocations(collection, num=10, frequencyThreshold=3, windows_size=2, filter_word=None):
    """
    This methods extracts, for each document collection, the top N bigram collocations. Bigram collocations
    are pairs of words which commonly co-occur. With a windows_size < 2, only bigrams formed by consecutive words
    will be taken into account. In that case, the result is consistent with a list of 2-words expressions that
    frequently appear in the collection. For windows_size > 2, all pairs of words within a windows of windows_size
    words will be considered. In that case, the result is consistent with a list of 2 related words that frequently
    co-occur together and therefore commonly have a semantic relationship between them
    """
    from nltk import collocations
    words = tokenize_collection(collection)
    bigram_measures = collocations.BigramAssocMeasures()
    if windows_size > 2:
        finder = collocations.BigramCollocationFinder.from_words(words,windows_size)
    else:
        finder = collocations.BigramCollocationFinder.from_words(words)

    finder.apply_freq_filter(frequencyThreshold)
    if filter_word:
        finder.apply_ngram_filter(lambda *w: filter_word not in w)
    return finder.nbest(bigram_measures.chi_sq, num)

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
    model = Word2Vec(documents, size=n_dim, window=5, min_count=3, workers=4)
    return model


def tokenize_collection(collection, lowercase=True, stopwords=True):
    documents = [tokenize_document(document, lowercase=lowercase, stopwords=stopwords) for document in collection]
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

    result = [token for token in tokens if not token in stops and len(token) >= min_length]
    return result


def replace_badwords(comment, badwords):
    comment = comment.lower()
    for badword in badwords:
        comment = comment.replace(badword, " fakeinsult ")
    return comment
