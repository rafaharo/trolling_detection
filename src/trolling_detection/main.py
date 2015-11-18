TRAIN_FILE = "resources/train/train.csv"
TEST_FILE = "resources/test/test_with_solutions.csv"
BAD_WORDS_FILE = "resources/badwords.txt"
INI_FILE = "resources/execution.ini"
NO_INSULT = 'NoInsult'
INSULT = 'Insult'
EXECUTION_SECTION = "Execution"
FEATURING_SECTION = "Featuring"

def preprocess_comment(comment):
    comment = comment.strip().strip('"')
    comment = comment.replace('_', ' ')
    comment = comment.replace('\n', ' ')
    comment = comment.replace("\\\\", "\\")
    return comment.decode('unicode-escape')


def load_csv_data(csv_file):
    import csv
    categories = []
    comments = []
    with open(csv_file, 'rb') as raw_dataset:
        dataset_reader = csv.reader(raw_dataset, delimiter=',')
        next(dataset_reader, None) # Filter Header
        for row in dataset_reader:
            categories.append(int(row[0]))
            comments.append(preprocess_comment(row[2]))
    return categories, comments

def load_bad_words(badwords_file):
    with open(badwords_file) as f:
        lines = f.readlines()
        return [badword[:-1] for badword in lines]


def prediction_info(pred, ground_truth):
    print "Accuracy: " + str(np.mean(pred == ground_truth)) + "\n"
    print(metrics.classification_report(ground_truth, pred,
        target_names=['NoInsult', 'Insult']))

def prediction_info_proba(pred, ground_truth):
    # Extract Plain Estimations

    labels = []
    for index, prediction in enumerate(pred):
        if prediction[0] > prediction[1]:
            labels.append(0)
        else:
            labels.append(1)

    labels = np.array(labels)

    print "Accuracy: " + str(np.mean(labels==ground_truth)) + "\n"
    print(metrics.classification_report(ground_truth, labels,
        target_names=['NoInsult', 'Insult']))

if __name__ == "__main__":
    import os
    from trolling_detection.train import train_basic
    from trolling_detection.train import naive_classifier
    from sklearn import metrics
    import numpy as np
    import ConfigParser

    current_dir = os.path.dirname(os.path.realpath(__file__))
    source_path = "src/trolling_detection"
    if source_path in current_dir:
        index_of = current_dir.index(source_path)
        path = current_dir[0:index_of]
    elif current_dir.endswith("trolling_detection") and not "src" in current_dir:
        path = current_dir

    badwords = load_bad_words(os.path.join(path, BAD_WORDS_FILE))
    categories, comments = load_csv_data(os.path.join(path, TRAIN_FILE))
    test_categories, test_comments = load_csv_data(os.path.join(path, TEST_FILE))
    assert len(categories) == len(comments) == 3947
    config_parser = ConfigParser.ConfigParser()
    config_parser.read(os.path.join(path, INI_FILE))

    fake = config_parser.getboolean(FEATURING_SECTION, 'replace_badwords')

    if fake:
        from feature_extraction import replace_badwords
        comments = [replace_badwords(comment, badwords) for comment in comments]
        test_comments = [replace_badwords(comment, badwords) for comment in test_comments]

    if config_parser.getboolean(EXECUTION_SECTION, 'naive'):
        naive_predictions = []
        for comment in test_comments:
            naive_predictions.append(naive_classifier(comment,  badwords, fake=fake))
        counter = 0
        for i, category in enumerate(test_categories):
            if category == 0 and naive_predictions[i] == 1:
                #print test_comments[i]
                counter += 1
        print counter
        naive_predictions = np.array(naive_predictions)
        print "\nNaive Model Result\n"
        prediction_info(naive_predictions, test_categories)

    if config_parser.getboolean(EXECUTION_SECTION, 'tf-idf'):
        classifier = train_basic(categories, comments)
        predictions = classifier.predict(test_comments)
        print "\nTF-IDF Model Result\n"
        prediction_info(predictions, test_categories)

    if config_parser.getboolean(EXECUTION_SECTION, 'custom'):
        from train import train_custom
        classifier = train_custom(categories, comments, badwords)
        predictions = classifier.predict(test_comments)
        print "\nCustom Model Result\n"
        prediction_info(predictions, test_categories)

    if config_parser.getboolean(EXECUTION_SECTION, 'average'):
        from train import train_assembling_average
        classifier = train_assembling_average(categories, comments, badwords)
        predictions = classifier.predict_proba(test_comments)
        print "\nAverage Ensemble Model Result\n"
        prediction_info_proba(predictions, test_categories)

    # Pruebas
    if config_parser.getboolean(EXECUTION_SECTION, 'collocations'):
        from feature_extraction import extract_top_bigrams_collocations
        from feature_extraction import similar_words
        from feature_extraction import language_model
        from feature_extraction import word2vec_model
        #collocations = extract_top_bigrams_collocations(comments, 10, windows_size=5,filter_word='fakeinsult')
        collocations = extract_top_bigrams_collocations(comments, 10, windows_size=5)
        print collocations
        similar_words(comments, "loser")
        model = language_model(comments)
        print model["are"].samples()
        model = word2vec_model(comments)
        print model.similarity('retarded', 'loser')

    if config_parser.getboolean(EXECUTION_SECTION, 'WordVec'):
        from train import train_word2vec
        from train import w2vectorize
        from feature_extraction import tokenize_document
        model, classifier = train_word2vec(categories, comments, 500)
        test_documents = [tokenize_document(document, stopwords='english') for document in test_comments]
        test_vecs = w2vectorize(test_documents, model, 500)
        predictions = classifier.predict(test_vecs)
        print "\nWord2Vec Model Result\n"
        prediction_info(predictions, test_categories)

    if config_parser.getboolean(EXECUTION_SECTION, 'Final'):
        from train import train_assembling
        classifier = train_assembling(categories, comments, badwords)
        predictions = classifier.predict(test_comments)
        print "\nFeature Ensemble Model Result\n"
        prediction_info(predictions, test_categories)