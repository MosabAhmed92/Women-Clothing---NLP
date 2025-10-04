import pickle
import os


def persist_vectorizer(vec, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')

    with open (path, 'wb') as f:
        pickle.dump(vec, f)
