def extract_top_bigrams_collocations(collection, num=10, frequencyThreshold=3, windows_size=5, filter_word=None):
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


def corpus_stats(collection):
    import nltk
    import pprint
    words = tokenize_collection(collection, lowercase=True, stopwords='english', min_length=3)
    text = nltk.Text(word.lower() for word in words)
    print "Number of Words: " + str(len(text))
    print "Number of unique words: " + str(len(set(text)))
    dist = nltk.FreqDist(text)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(dist.most_common(20))
    print dist['stupid']


def tf_idf_stats(collection, num=20):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(analyzer='word', min_df=4, stop_words = 'english')
    matrix = tfidf.fit_transform(collection)
    dense = matrix.todense()
    features_names = tfidf.get_feature_names()
    print "\nNumber of Features: " + str(len(features_names))
    for index, row in enumerate(dense[0:num]):
        print "\nScores for comment: " + str(index)
        comment = row.tolist()[0]
        scores = [pair for pair in zip(range(0, len(comment)), comment) if pair[1] > 0]
        sorted_scores = sorted(scores, key=lambda t: t[1] * -1)
        for phrase, score in [(features_names[word_id], score) for (word_id, score) in sorted_scores][:num]:
            print('{0: <20} {1}'.format(phrase, score))
        print "\n"





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


def tokenize_collection(collection, lowercase=True, stopwords=True, min_length=3):
    documents = [tokenize_document(document, lowercase=lowercase, stopwords=stopwords, min_length=min_length) for document in collection]
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
        if badword is not 'fakeinsult':
            comment = comment.replace(badword, " fakeinsult ")
    return comment

def collection_text_similarity(collection, tokenizer=None, ngram_range=(1, 1)):
    """
    This function calculate the cosine based similarity between each pair of documents in the collection
    First, the TF-IDF transformation of the documents is performed to later calculate the similarity
    matrix by calculating the cosine between the vectors of tokens

    :param collection: Collection of documents (list of strings)
    :type collection: list
    :param tokenizer: [Optional] Tokenizer to be applied to the texts
    :type tokenizer: sandbox.document_processing.categorizer.tokenizer.Tokenizer
    :param ngram_range: [Optional] Ngrams Range
    :type ngram_range: list
    :return: Collection Similarity Matrix
    :rtype:
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(decode_error='replace',
                                 strip_accents='unicode',
                                 lowercase=True,
                                 tokenizer=tokenizer, ngram_range=ngram_range)
    tfidf = vectorizer.fit_transform(collection)
    return (tfidf * tfidf.T).A
