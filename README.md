

### Code Overview

- `loader.py`: import data from .csv and create a target variable.
- `processor.py`: preprocess predictor variables and create new variables from existing ones. 
- `model_builder.py`: perform grid search with stratified k fold.
- `utils.py`: utility functions to run a benchmark, print results, and save/load models.
- `main.py`: helper workflow script, which ties everything together. 
- `tests folder`: unit tests for Loader and Processor classes.

### Data Description

#### Target variable

Column 'class' is used as target variable. The original values are mapped to either 'true' or 'false'. This mapping was 
chosen after a brief inspection of the verifier site criteria (where available, also see the `reduce_class_to_binary` 
method of the `Loader` class).

86 articles were discarded as they are assigned a class value for which was not possible to determine a mapping to true 
or false.

#### Predictors

The following predictors are used to represent each article:
- `country1`: one of the countries the news article was published in (as a categorical).
- `number_regions_published`: sum of non-empty values in country1, country2, country3, and country4 (numeric). 
- `published_date_days_diff`: number of days since the earliest date in published_date (numeric).
- `lang`: language code (as categorical).
- `ref_source`: referenced by (as categorical).
- `verifiedby`: fact checker site (as categorical).

For all categoricals, categories with less than 10 examples are lumped under an 'other' category.

In addition, the original column 'title' can be used in one of two ways:
1. After lowercasing, removing punctuation, stop words, digits, lemmatizing, build a tf-idf matrix on the bag-of-words 
   of the titles. The matrix is used alongside the predictors above to train models.
2. Each title is represented by a 300-dimensional vector, provided by spaCy's Doc.vector property (this is the average 
   of the title's token embedding vectors). The vectors are used alongisde the predictors above to train models.  

Note that titles for which `verifiedby` is equal to `snopes` contain information about the target variable. To mitigate 
(and prevent the models from 'cheating'), all titles from snopes are replaced with the respective value of 
`source_title` prior to preprocessing.

#### Unused Columns

The following columns in the original data set were not used in the modelling workflow:
- `country2`, `country3`, `country4`: very sparse (but used to create the 'number_regions_published' variable)
- `country`: part of it is explicit in country1 (which is used)
- `article_source`: urls, which contain info implicit in verifiedby
- `category`: very sparse.
- `source_title`: title preferred over source_title as one of the preprocessing steps consists in lemmatization (which may
  not be available for all languages)
- `content_text`: seems some examples contain target variable information.

### Methodology

#### Evaluation metric

The percentage of observations classified as 'true' is about 1% of the total. This makes the data set very unbalanced 
wrt the target, thus macro-F1 score the natural choice to evaluate the models (as it will give equal 'weight' to both 
classes, regardless of their size).

#### Model Search

Prior to the cross-validation, the data set is split into a train and hold out set (80/20) split. The training portion 
is then used to perform a grid search with multiple models, using a stratified k-fold strategy. The goal of this grid 
search step is to determine the best model (that is, the classifier and set of parameter values that reaches the 
highest macro-F1 score).  

Note that the best model is evaluated on the holdout set only after cross validation is complete.

#### Results

The best model is obtained for the Multinomial Naive-Bayes classifier (with an alpha parameter value of 0.95), using 
the title variable represented as a tf-idf matrix. The holdout set macro-F1 score is 0.686.

To print out details of the best model built with the approach described in the previous sections, run 
`python main.py show 'best_model_Dec-31-2020_1529.pkl'`

To reproduce the entire modelling workflow, run `python main.py runall`

To add more classifiers/parameters to the search, modify the `benchmark` function in `utils.py`.

#### Possible Next Steps

- improve mapping of 'class' values to true/false.
- add columns to data set:
    - more specifically, add the original title and content of a potential fake news article, instead of just the title 
      and content of the verifier.
    - aim to add more news articles deemed to be 'true'.
- improve test coverage.
- add classifiers and increase parameter search space. 
- implement a deep learning classifier, e.g., LSTM-based:
    - take advantage of the token vectors provided by spaCy to create a sequential document representation suitable for
      a LSTM input. 