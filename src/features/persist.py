import pickle
import os


def save_vectorizer(vec, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')

    with open (path, 'wb') as f:
        pickle.dump(vec, f)


def load_vectorizer(path, name):
    file_path = os.path.join(path, f"{name}.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No Vecotrizer Found on {file_path}")
    with open(file_path, 'rb') as f :
        vectorizer = pickle.load(f)

    return vectorizer

