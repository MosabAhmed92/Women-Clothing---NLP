"""Checking if the Features made by TFIDF actually carry a signal"""

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.eval.metrics import evaluate_model




def train_dummy(X_tr, y_tr, X_val, y_val):
    model = DummyClassifier(strategy= 'most_frequent')
    metrics = evaluate_model(model, X_tr, y_tr, X_val, y_val, 'Dummy_Classifier')
    return model, metrics
 
def train_nb(X_tr, y_tr, X_val, y_val):
    model = MultinomialNB(alpha = 1.0)
    metrics = evaluate_model(model, X_tr, y_tr, X_val, y_val, 'Naive-Bayes Classifier')
    return model, metrics

def train_log(X_tr, y_tr, X_val, y_val):
    model = LogisticRegression(C= 1.0, solver='liblinear', max_iter=10000, multi_class='ovr', n_jobs=-1)
    metrics = evaluate_model(model, X_tr, y_tr, X_val, y_val, 'Logistic_Regression')
    return model, metrics

def train_svc(X_tr, y_tr, X_val, y_val):
    model = LinearSVC(C = 1.0, class_weight=None)
    metrics = evaluate_model(model, X_tr, y_tr, X_val, y_val, 'Support Vector Classifier')
    return model, metrics



def run_all_baselines(X_tr, y_tr, X_val, y_val):
    """
    Train/evaluate all baseline models and return:
      - models: dict[str, fitted_model]
      - results: list[metrics_dict] sorted by 'macro_f1_val' DESC
    """
    models = {}
    results = []

    m, r = train_dummy(X_tr, y_tr, X_val, y_val)
    models['Dummy'] = m ; results.append(r)


    m, r = train_nb(X_tr, y_tr, X_val, y_val)
    models['Naive_Bayes'] = m ; results.append(r)


    m, r = train_log(X_tr, y_tr, X_val, y_val)
    models['Logistic_regression'] = m ; results.append(r)

    m, r = train_svc(X_tr, y_tr, X_val, y_val)
    models['Support_Vector_Classification'] = m ; results.append(r)

    return models, results