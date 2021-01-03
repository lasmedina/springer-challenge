### Challenge Solution Overview

The original .csv consists of 6902 rows and 15 columns. The goal is to create a binary classifier, using the `class` 
column as the target variable, to determine whether a news article is "fake". Most of the columns carry information 
about the fake news verifier website which verifies the article (and not necessarily about the original news article 
itself).

#### Target variable

The original values of `class` must be mapped to either `true` or `false`. This mapping was chosen after a brief 
inspection of the criteria used by the verifier websites (where available, also see the `reduce_class_to_binary` method 
of the `Loader` class).

86 articles are discarded as they were assigned a `class` value for which is not possible to determine an unambiguous 
mapping to either `true` or `false`.

#### Predictors

The following predictors are used to represent each article:
- `country1`: one of the countries the news article was published in (as a categorical).
- `number_regions_published`: sum of non-empty values in country1, country2, country3, and country4 (numeric). 
- `published_date_days_diff`: number of days since the earliest date in published_date (numeric).
- `lang`: new's source language code (as categorical).
- `ref_source`: referenced source (as categorical).
- `verifiedby`: fact checker site (as categorical).

For all categoricals, categories with less than 10 examples are lumped under a category named `other`.

In addition, the original column 'title' can be used in one of two ways:
1. Build a tf-idf matrix on the bag-of-words of the titles (After lower casing, removing punctuation, stop words, 
   digits, and lemmatizing the tokens in the titles). The matrix is used alongside the predictors above to train models.
2. Each title is represented by a 300-dimensional vector, provided by spaCy's `Doc.vector` property (this is the average 
   of the title's token embedding vectors). The vectors are used together with the predictors above to train models.  

**_Note:_** titles for which `verifiedby` is equal to `snopes` contain information about the target variable. To mitigate 
(and prevent the models from 'cheating'), all titles verified by snopes are replaced with the respective value of 
`source_title` prior to preprocessing.

#### Unused Columns

The following columns in the original data set were not used in the modelling workflow:
- `country2`, `country3`, `country4`: very sparse (but used to create the `number_regions_published` predicto)
- `country`: part of it is explicit in `country1` (which is used)
- `article_source`: urls, which contain info implicit in `verifiedby`
- `category`: very sparse.
- `source_title`: the column is used instead, as one of the preprocessing steps consists in lemmatization (which 
  may not be available for all languages). However, source_title is used in place of title for articles verified by 
  snopes (see note above).
- `content_text`: to be considered in a future iteration of this solution.

### Methodology

The following sections describe the methodology used to train and evaluate classifiers using the predictors and target
variables outlined above.

#### Evaluation Metric

The percentage of observations classified as `true` is about 1% of the total. This makes the data set very unbalanced 
wrt the target, thus macro-F1 score is the natural choice to evaluate the models (as it will give equal 'weight' to both 
classes, regardless of their size).

#### Model Search

Prior to the cross-validation, the data set is split into a train and hold out set (80/20) split. The training portion 
is then used to perform a grid search with multiple models (as well as with a baseline classifier), using a stratified 
k-fold strategy. The goal of this grid search step is to determine the best model, that is, the classifier and set of 
parameter values that reaches the highest macro-F1 score during k-fold training.

The best model is then evaluated on the holdout set (only after cross validation with all classifiers is complete).

The best model and respective metrics are stored in a `.pkl`, under the `data` folder.

#### Results

The model search was ran for two different sets of predictor variables (see the Predictors section above). 
The difference between the two is the representation of `title` (either a tf-idf matrix or averaged word vectors).

The best model was obtained for the Multinomial Naive-Bayes classifier (with an alpha parameter value of 0.95), using 
the `title` variable represented as a tf-idf matrix. The holdout set macro-F1 score of this model is 0.686.

### Code

High-level summary of code organization below:
- `loader.py`: import data from .csv and create a target variable.
- `processor.py`: preprocess predictor variables and create new variables from existing ones. 
- `model_builder.py`: perform grid search with stratified k-fold.
- `utils.py`: utility functions to run a benchmark, print results, and save/load models.
- `main.py`: helper workflow script, which ties everything together. 
- `tests folder`: unit tests for Loader and Processor classes.

To print out details of the best model built with the approach described in the previous sections, run 
`python main.py show 'best_model_Dec-31-2020_1529.pkl'`

To reproduce the entire modelling workflow used to create the best classifier described above, run 
`python main.py run tfidf` (this will take a couple of minutes to run and will store the results in a `.pkl` 
under `data`).

To reproduce the modelling workflow using the averaged word vectors for title representation, run 
`python main.py run emb` 

To add more classifiers/parameters to the search, modify the `benchmark` function in `utils.py`.

### Possible Next Steps

- Improve/confirm mapping of 'class' values to true/false.
- Add columns to data set:
    - more specifically, add the original title and content of a potential fake news article, instead of just the title 
      and content of the verifier website.
    - aim to add more news articles deemed to be 'true'.
- Improve test coverage.
- Add classifiers and increase parameter search space. 
- Implement a deep learning classifier, e.g., LSTM-based:
    - take advantage of the token vectors provided by spaCy to create a sequential document representation suitable for
      a LSTM input. 