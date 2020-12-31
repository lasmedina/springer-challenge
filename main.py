import argparse
import sys
from pathlib import Path

from src.loader import Loader
from src.processor import Processor

from src.utils import grid_search_with_selected_preds, load_model


def summarize_best_model(filename: str):
    contents = load_model(filename)
    for k, v in contents.items():
        print(k, ' ---> ', v)


def run_complete_workflow(use_tfidf_on_title: bool):
    print('Loading and cleaning data...')
    data_folder = Path("data")
    file_loc = data_folder / "dataset.csv"

    loader = Loader(file_loc)
    loader.load_data()
    loader.reduce_class_to_binary()
    loader.clean_data()

    print('Class summary:')
    print(loader.df['class'].value_counts())

    # Preprocess existing predictors and create some new ones.
    print("")
    print('Preprocessing predictors...\n')
    processor = Processor(loader.df)
    processor.preprocess_title()
    processor.preprocess_categorical_column(['lang', 'verifiedby', 'ref_source', 'country1'])
    processor.create_day_diff_variable(['published_date'])
    processor.create_number_regions(['country1', 'country2', 'country3', 'country4'])
    if not use_tfidf_on_title:
        # Compute averaged embedding title vector.
        processor.create_title_vector()

    # Drop unused columns from dataframe.
    processed_df = processor.df
    processed_df = processed_df.drop(['country', 'published_date', 'country2', 'country3', 'country4',
                                      'article_source', 'source_title', 'content_text', 'category'], axis=1)

    # Perform grid search and evaluate models with stratified k-fold cross validation.
    # Specify the df columns to be used as predictors.
    cat_columns = ['lang', 'verifiedby', 'ref_source', 'country1']
    num_columns = ['number_regions_published', 'published_date_day_diff']

    print("Building and evaluating models....")

    if use_tfidf_on_title:
        # Set of models trained using the title represented as a tf-idf on a bag of words.
        print("Title represented as a tf-idf matrix on a bag of word counts...")
    else:
        print("Title represented as a vector (averaged across individual word embeddings)...")

    grid_search_with_selected_preds(processed_df, cat_columns, num_columns, use_tfidf_on_title)


if __name__ == "__main__":

    opt = ""
    use_tfidf_on_title = True
    mdl_file = 'best_model_Dec-31-2020_1529.pkl'
    if len(sys.argv) <= 1 or sys.argv[1] not in ('runall', 'show'):
        raise argparse.ArgumentTypeError("Option must be 'show' or 'runall'.")
    opt = sys.argv[1]

    if len(sys.argv) > 2:
        use_tfidf_on_title = bool(sys.argv[2])
    if len(sys.argv) > 3:
        mdl_file = sys.argv[3]

    if opt == "show":
        summarize_best_model(mdl_file)
    elif opt == "runall":
        run_complete_workflow(use_tfidf_on_title)
