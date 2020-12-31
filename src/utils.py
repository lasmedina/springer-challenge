import time
from pathlib import Path
from typing import List, Dict

import joblib
from pandas import DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.model_builder import ModelBuilder
import numpy as np

data_folder = Path("data")


def save_model(mdl_obj: object, mdl_predictors: Dict, best_score: float, test_score: float):
    # Save sklearn model and predictors used to train it on a .pkl
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    file_name = ('best_model_' + timestamp + '.pkl')
    mdl_file = data_folder / file_name

    mdl_dict = {'Model details': mdl_obj,
                'Predictor names': mdl_predictors,
                'Best cv macro-F1': best_score,
                'Macro-F1 on test set': test_score}

    joblib.dump(mdl_dict, mdl_file)


def load_model(mdl_file: str):
    # Load previously saved model and predictors.
    contents = joblib.load(data_folder / mdl_file)
    return contents


def benchmark(mb: ModelBuilder) -> (GridSearchCV, str, float, Dict[str, List[str]]):
    # Call grid search for multiple classifiers and parameters.
    # Print best score and parameter combination for each classifier.
    # Return best model.

    clfs = [MultinomialNB(),
            ComplementNB(),
            SGDClassifier(random_state=0),
            SVC(random_state=0),
            KNeighborsClassifier()
            ]
    params_list = [{'classifier__alpha': np.arange(0.1, 1.0, 0.05)},
                   {'classifier__alpha': np.arange(0.1, 1.0, 0.05)},
                   {'classifier__loss': ['hinge', 'log', 'perceptron', 'squared_hinge'],
                    'classifier__alpha': [0.0001, 0.001, 0.002, 0.01], 'classifier__max_iter': [1000, 2000]},
                   {'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'classifier__degree': [2, 3, 4],
                    'classifier__C': [1.0, 0.9]},
                   {'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance'],
                    'classifier__p': [1, 2]}
                   ]

    best_score = 0
    best_params = {}
    best_clf = []
    best_clf_name = ""
    for i in range(len(clfs)):
        trained_clf, clf_score, clf_params = mb.do_cv(clfs[i], params_list[i])
        print("\tBest score for", clfs[i], " with parameters", clf_params, ":", clf_score)
        if clf_score > best_score:
            best_clf = trained_clf
            best_score = clf_score
            best_params = clf_params
            best_clf_name = clfs[i].__class__.__name__

    return best_clf, best_clf_name, best_score, best_params


def grid_search_with_selected_preds(df: DataFrame, cat_columns: List[str], num_columns: List[str],
                                    use_tfidf_on_text: bool):
    # Helper function to run grid search using predefined predictors.

    m_builder = ModelBuilder(df, cat_columns, num_columns, use_tfidf_on_text)

    # First, compute a baseline for the selected predictors.
    # Hopefully all the models we train will perform better than it.
    _, score, _ = m_builder.do_cv(DummyClassifier(random_state=0),
                                  param_grid={'classifier__strategy': ['stratified']})
    print("Baseline model (DummyClassifier) macro-F1: ", score)
    print("")

    # Cross-validate multiple models with different parameter combinations and output the best score and parameters for 
    # each estimator.
    print("Running grid search for multiple classifiers....")
    best_grid_cv_obj, best_clf_name, best_score, best_params = benchmark(m_builder)
    print("")
    print("Best overall model: ")
    print("\t - Classifier: ", best_clf_name)
    print("\t - Macro F1-score: ", best_score)
    print("\t - Classifier parameters: ", best_params)
    print("")

    # Evaluate the best overall model on a holdout set.
    test_score = m_builder.evaluate_on_holdout(best_grid_cv_obj)
    print("Macro-F1 score of best model on holdout set:", test_score)

    # Save the best model and the predictors used in this grid search.
    vars_dict = {'cat_columns': cat_columns, 'num_columns': num_columns, 'use_tfidf_on_title': use_tfidf_on_text}
    save_model(best_grid_cv_obj.best_estimator_, vars_dict, best_score, test_score)
