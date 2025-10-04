"""Vectorizer factory for BoW/TF-IDF; exposes fit/transform stubs."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def build_vectorizer(cfg):
    vectorizer_type = cfg['vectorization']['type']
    ngram_range = tuple(cfg['vectorization']['ngram_range'])  
    print('Vectorizer type from config is: ', repr(vectorizer_type))


    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_features=cfg['vectorization']['max_features'],
            ngram_range= ngram_range,
            min_df=cfg['vectorization']['min_df']
        )
    elif vectorizer_type == 'bow':
        vectorizer = CountVectorizer(
            max_features=cfg['vectorization']['max_features'],
            ngram_range=ngram_range,
            min_df=cfg['vectorization']['min_df']
        )
    else:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

    return vectorizer

def fit_vectorizer (cfg, text):
    vec = build_vectorizer(cfg)
    X_train = vec.fit_transform(text)
    return X_train, cfg


def transform_text(text, vectorizer):
    X = vectorizer.transform(text)

    return X











