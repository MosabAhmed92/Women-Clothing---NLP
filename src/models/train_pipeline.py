from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pickle import dump
import numpy as np 
import yaml

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def run_gridsearch(X_train, y_train, random_state = 42):

    # Define the Scorer and the pipline
    scorer = make_scorer(f1_score, average = 'macro')

    pipe = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LogisticRegression())
                     ])
    # Defining Parameters Grid
    grid = {
        "tfidf__ngram_range": [(1,1),(1,2)],
        "tfidf__min_df": [1,2,5],
        "tfidf__max_df": [0.9,0.95],
        "tfidf__sublinear_tf": [True],
        "clf": [LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)],
        "clf__C": [0.5,1,2,3,5],
        "clf__class_weight": [None, "balanced"],
    }

    # Cross_validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(pipe, grid, scoring=scorer, cv = cv, n_jobs=-1, verbose=2, refit= True)

    # fitting
    gs.fit(X_train, y_train)

    print("Best Macro-F1 :", gs.best_score_)
    print("Best Parameters :", gs.best_params_)

    return gs.best_estimator_    


