import unittest
import spacy
from numpy import NaN

from src.processor import get_processed_tokens
from src.processor import Processor
from pandas import DataFrame

nlp = spacy.load('en_core_web_sm')


class TestProcessor(unittest.TestCase):

    def test_get_processed_tokens(self):
        # assert no punctuation, assert no stopwords, assert lemmas
        txt = 'This sentence has some words.'
        doc = nlp(txt)

        expected = 'this sentence have some word'

        actual = get_processed_tokens(doc)
        self.assertEqual(expected, actual)

    def test_preprocess_title(self):
        title = ["This Title has some words.", "the mice ate the cheese!!!"]
        verifiedby = ["source 1", "source 2"]
        source_title = title
        df = DataFrame(list(zip(title, verifiedby, source_title)),
                       columns=['title', 'verifiedby', 'source_title'])

        proc = Processor(df)
        proc.preprocess_title()
        actual_df = proc.df

        expected_title = ["this title have some word", "the mouse eat the cheese"]
        expected_df = DataFrame(list(zip(expected_title, verifiedby, source_title)),
                                columns=['title', 'verifiedby', 'source_title'])

        self.assertEqual(list(expected_df['title']), list(actual_df['title']))
        self.assertEqual(list(expected_df['verifiedby']), list(actual_df['verifiedby']))
        self.assertEqual(list(expected_df['source_title']), list(actual_df['source_title']))

    def test_replace_snopes_title(self):
        title = [
            "Does ‘Every Election Year’ Have a Coinciding Disease? A conspiratorial meme about disease outbreaks got "
            "a number of key facts wrong. False",
            "Police asked shops selling alcohol to restrict opening hours on St. Patrick’s Day to lower the risk of "
            "house parties where coronavirus could be spread."]
        verifiedby = ["snopes",
                      " TheJournal.ie"]
        source_title = ["Does 'Every Election Year' Have a Coinciding Disease?",
                        "Debunked: No, gardaí did not ask off-licences to open late on St Patrick's Day to stop house "
                        "parties"]
        df = DataFrame(list(zip(title, verifiedby, source_title)),
                       columns=['title', 'verifiedby', 'source_title'])

        proc = Processor(df)
        proc.replace_snopes_titles()
        actual_df = proc.df

        expected_title = [source_title[0], title[1]]  # snopes title is replaced, while the other is unchanged.
        expected_df = DataFrame(list(zip(expected_title, verifiedby, source_title)),
                                columns=['title', 'verifiedby', 'source_title'])

        self.assertEqual(list(expected_df['title']), list(actual_df['title']))
        self.assertEqual(list(expected_df['verifiedby']), list(actual_df['verifiedby']))
        self.assertEqual(list(expected_df['source_title']), list(actual_df['source_title']))

    def test_preprocess_categorical_column(self):
        cat1 = ["A", "a", "a", "A", "a", "a", "a", "a", "a", "a", "a",
                "b", "b", "c", "d", "e"] + [NaN] * 10
        df = DataFrame(cat1, columns=['cat1'])

        proc = Processor(df)
        proc.preprocess_categorical_column(df.columns)
        actual_cat1 = list(proc.df['cat1'])

        expected_cat1 = ["a"] * 11 + ["other"] * 5 + ["unknown"] * 10

        self.assertEqual(expected_cat1, actual_cat1)
