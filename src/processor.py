from operator import index
from typing import List

import pandas
from pandas import DataFrame
import spacy
from spacy.tokens.doc import Doc


def get_processed_tokens(doc: Doc):
    # Replace with lemmatized tokens and remove punctuation, stop words, and digits.
    processed = [token.lemma_ for token in doc if not (token.is_punct | token.is_digit | token.is_stop)]
    return ' '.join(processed)


class Processor:
    """Processor class."""
    def __init__(self, df: DataFrame):
        self.df = df

    def replace_snopes_titles(self):
        # Snopes titles contain class information, must be replaced with respective source titles.
        self.df.title[self.df.verifiedby == 'snopes'] = self.df.source_title

    def preprocess_title(self):
        # Preprocess title column.
        self.replace_snopes_titles()

        titles = self.df['title'].tolist()
        titles = [title.lower().strip() for title in titles]

        nlp = spacy.load('en_core_web_md')

        processed_titles = []
        for text in nlp.pipe(titles):
            processed_titles.append(get_processed_tokens(text))

        self.df['title'] = processed_titles

    def preprocess_categorical_column(self, col_names: index):
        # Columns to be treated as categoricals have all categories with less than 10 examples merged.
        # Making bold assumption that category 'XYZ' is the same as category 'xyz' or 'xYz'.
        for col_name in col_names:
            self.df[col_name] = self.df[col_name].fillna("unknown")
            self.df[col_name] = self.df[col_name].str.lower()
            self.df.loc[self.df[col_name].value_counts()[self.df[col_name]].values < 10, col_name] = "other"

            processed_col = self.df[col_name].astype('category')
            self.df[col_name] = processed_col

    def create_day_diff_variable(self, col_names: List[str]):
        # Create new time variable counting the number of days from earliest date.
        for col_name in col_names:
            self.df[col_name] = self.df[col_name].astype('datetime64[ns]')
            self.df[col_name + '_day_diff'] = (self.df[col_name] - self.df[col_name].min()).dt.days

    def create_number_regions(self, col_names: List[str]):
        # New numeric variable to count the number of regions where article was published.
        self.df['number_regions_published'] = len(col_names) - self.df[col_names].isna().sum(axis=1)

    def create_title_vector(self):
        # Vector representation for the title: each title vector is the average of individual word embeddings.
        titles = self.df['title'].tolist()

        nlp = spacy.load('en_core_web_md')

        processed_titles = []
        for doc in nlp.pipe(titles):
            title_vector = doc.vector
            processed_titles.append(title_vector)
        new_df = DataFrame(processed_titles)
        new_df.columns = ['doc_vec_' + str(i) for i in range(new_df.shape[1])]

        self.df.reset_index(drop=True, inplace=True)
        new_df.reset_index(drop=True, inplace=True)
        self.df = pandas.concat([self.df, new_df], axis=1)
