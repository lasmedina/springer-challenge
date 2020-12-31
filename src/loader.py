from pandas import read_csv
from pandas import DataFrame


class Loader:
    """Loader class."""

    def __init__(self, filepath: str):
        self.df = DataFrame
        self.filepath = filepath

    def load_data(self):
        # Import data from .csv as DataFrame
        col_names = ['verifiedby', 'country', 'class', 'title', 'published_date', 'country1',
                     'country2', 'country3', 'country4', 'article_source', 'ref_source',
                     'source_title', 'content_text', 'category', 'lang']
        col_types = {'title': str}
        df = read_csv(self.filepath, names=col_names, header=0, dtype=col_types)

        self.df = df

    def reduce_class_to_binary(self):
        # Rename values in class variable to either true or false.
        self.df['class'] = self.df['class'].str.lower()

        # Some values are ambiguous, replaced with 'unclear'.
        replacements = {
            'misleading': 'false',
            'mostly false': 'false',
            # see https://www.snopes.com/fact-check-ratings/ or
            # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
            'partly false': 'false',
            'no evidence': 'false',
            'explanatory': 'unclear',
            'mostly true': 'true',  # see https://www.snopes.com/fact-check-ratings/
            'half true': 'true',
            # see https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
            'unproven': 'unclear',  # https://www.snopes.com/fact-check-ratings/
            '(org. doesn\'t apply rating)': 'unclear',  # all articles are false except one, which is a Q&A
            'labeled satire': 'false',
            'two pinocchios': 'false',
            'partially false': 'false',  # see https://www.snopes.com/fact-check-ratings/
            'partly true': 'true',
            'scam': 'false',
            'fake': 'false',
            'correct': 'true',
            'misleading/false': 'false',
            'partially correct': 'true',
            'collections': 'unclear',  # snopes
            'unlikely': 'unclear',
            'fake news': 'false',
            'half truth': 'true',
            'false and misleading': 'false',
            'true but': 'true',
            'pants on fire': 'false',
            # https://www.politifact.com/article/2018/feb/12/principles-truth-o-meter-politifacts-methodology-i/
            'misinformation / conspiracy theory': 'false',
            'partially true': 'true',
            'not true': 'false',
            'unverified': 'unclear'
        }

        # Rename class values, drop rows where class is ambiguous, and convert class to category.
        self.df['class'] = self.df['class'].replace(replacements)
        self.df = self.df[self.df['class'] != 'unclear']

        df_copy = self.df.copy()
        df_copy['class'] = self.df['class'].astype('category')

        self.df = df_copy

    def clean_data(self):
        # Remove duplicates and rows with no class value.
        self.df = self.df.drop_duplicates(keep='first')
        self.df = self.df[self.df['class'].notna()]
