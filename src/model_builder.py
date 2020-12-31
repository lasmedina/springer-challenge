from typing import List, Dict

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from pandas import DataFrame


class ModelBuilder:
    """Model Builder class"""

    def __init__(self, df: DataFrame,
                 cat_columns: List[str],
                 num_columns: List[str],
                 use_tfifd_on_title=True):

        x = df.loc[:, df.columns != 'class']
        y = df['class'].to_numpy()

        self.skf = StratifiedKFold(n_splits=5,
                                   shuffle=True,
                                   random_state=0)

        if use_tfifd_on_title:  # build tfidf on title's bag of words.
            self.col_trans = ColumnTransformer(
                [("bow", TfidfVectorizer(min_df=9, ngram_range=(1, 1)), 'title'),
                 ("cat", OneHotEncoder(), cat_columns),
                 ("num", MinMaxScaler(), num_columns)
                 ],
                remainder='drop')
        else:  # use the title vector representation instead
            x = x.drop(['title'], axis=1)
            self.col_trans = ColumnTransformer(
                [("cat", OneHotEncoder(), cat_columns),
                 ("num", MinMaxScaler(), num_columns)
                 ],
                remainder=MinMaxScaler())  # need this preprocessor because some estimators cannot handle negative
            # values

        self.X_train, self.X_holdout, self.y_train, self.y_holdout = train_test_split(
            x, y, test_size=0.20, random_state=0, stratify=y)

    def do_cv(self, clf, param_grid: Dict) -> (GridSearchCV, float, Dict):
        # Perform grid search on stratified k fold.
        pipe = Pipeline(steps=[('preprocessor', self.col_trans),
                               ('classifier', clf)])
        param_grid = param_grid
        grid_search = GridSearchCV(pipe,
                                   param_grid=param_grid,
                                   cv=self.skf.split(self.X_train, self.y_train),
                                   scoring='f1_macro')
        best_clf = grid_search.fit(self.X_train, self.y_train)

        best_score = grid_search.best_score_
        best_params = grid_search.best_params_

        return best_clf, round(best_score, 3), best_params

    def evaluate_on_holdout(self, clf, holdout_set=None) -> float:
        # Evaluate how the model performs on a holdout set.
        if holdout_set is None:
            holdout_set = (self.X_holdout, self.y_holdout)

        predictions = clf.predict(holdout_set[0])
        test_score = f1_score(holdout_set[1], predictions, average='macro')

        return round(test_score, 3)
