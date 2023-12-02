def load_stopwords(language, source_path='../data/stopwords/'):
    stopwords = []
    with open(f'{source_path}{language}.txt', 'r') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords
