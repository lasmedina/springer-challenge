import unittest

from src.loader import Loader

local_path = '~/PycharmProjects/springerChallenge/data/dataset.csv'
loader = Loader(local_path)


class TestLoader(unittest.TestCase):

    def test_original_shape(self):
        expected_shape = (6902, 15)
        loader.load_data()

        self.assertEqual(loader.df.shape, expected_shape)

    def test_column_names(self):
        expected_names = ['verifiedby', 'country', 'class', 'title', 'published_date', 'country1',
                          'country2', 'country3', 'country4', 'article_source', 'ref_source',
                          'source_title', 'content_text', 'category', 'lang']
        loader.load_data()
        self.assertEqual(list(loader.df.columns), expected_names)

    def test_class_dtype(self):
        loader.load_data()
        loader.reduce_class_to_binary()

        self.assertEqual(loader.df['class'].dtype, 'category')

    def test_class_is_binary(self):
        expected_cats = ['false', 'true']

        loader.load_data()
        loader.reduce_class_to_binary()
        actual_cats = list(loader.df['class'].cat.categories)

        self.assertEqual(actual_cats, expected_cats)

    def test_no_duplicates(self):
        loader.load_data()
        loader.reduce_class_to_binary()
        loader.clean_data()

        clean_dups = loader.df
        clean_dups_shape = clean_dups.loc[clean_dups.duplicated(), :].shape

        expected_dup_shape = (0, len(loader.df.columns))
        self.assertEqual(clean_dups_shape, expected_dup_shape)
