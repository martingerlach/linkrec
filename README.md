# research/mwaddlink

This is the repository that backs the [Wikimedia Link Recommendation service](https://wikitech.wikimedia.org/wiki/Add_Link).
It contains code for training a model and generating datasets, as well as an HTTP API and command line interface for
fetching link recommendations for Wikipedia articles.

The method is context-free and can be scaled to (virtually) any language, provided that we have enough existing links
to learn from.

## Requirements

- You need set up a python virtual environment and install the packages from the requirements-file:

```bash
virtualenv -p /usr/bin/python3 venv
source venv_query/bin/activate
pip install -r requirements-query.txt
```
- on the stat-machines, make sure you have the http-proxy set up https://wikitech.wikimedia.org/wiki/HTTP_proxy
- you might have to install the following nltk-package manually: ```python -m nltk.downloader punkt```

## Training the model

There is a pipeline to train the full model

```bash
WIKI_ID=enwiki ./run-pipeline.sh
```

**Notes**:
- some parts in the script rely on using the spark cluster using a specific conda-environment from a specific stat-machine (stat1008).
- on the stat-machines, make sure you have the http-proxy set up https://wikitech.wikimedia.org/wiki/HTTP_proxy
- you might have to install the following nltk-package manually: ```python -m nltk.downloader punkt```

**The pipeline executes the following steps**

Pre-processing:
- generate an anchor dictionary
  - generate_anchor_dictionary_spark.py parses the wikitext-dump and extracts anchors (anchors.pkl) all redirects (redirects.pkl) and all pages in the main namespace containing the mapping page-title to page-id (pageids.pkl)
  - generate_wdproperties_spark.py parses the wikidata-dump and extracts all statements for the instance-of property (P31) of the main-namespace articles in a wiki. this yields a dictionary with page-id and a list of all P-31 statements (wdproperties.pkl)
  - filter_dict_anchor.py filters the anchor-dictionary by removing all links to pages belonging to a given class of entities (e.g. disambiguation pages, list pages, etc)

- generate entity embedding
  - running wikipedia2vec on a wikipedia-dump to generate embeddings
  - filter_dict_w2v.py filters the embedding to only main namespace articles contained in the pageids.pkl dictionary.
  - final embedding: w2vfiltered.pkl

- generate navigation embedding
  - generate wikidata-embedding (this has to be done only once for all wikis)
    - generate_features_nav2vec-01-get-sessions_wikidata.py gets reading sessions of wikidata-items aggregated for all wikis
    - generate_features_nav2vec-02-train-w2v.py uses fasttext to train an embedding of wikidata-items based on reading sessions
  - projecting the wikidata-embedding into a given wiki
    - generate_features_nav2vec-03-get-qids_wikidata.py gets the mapping of pageids (article-ids in a wiki) to qids (wikidata-item ids) yielding qids.pkl
    - filter_dict_nav_wikidata.py filters the wikidata-embedding to wikidata-items which have an article in the given wiki
  - final embedding: navfiltered.pkl

- extract training and testing sentences
  - generate_backtesting_data.py parses the dumps to extract 200,000 sentences (50-50 split into training and test) each from a different article extracting the first sentence with at least one link. yields training/sentences_train.csv and testing/sentences_test.csv

- generate training features
  - generate_training_data.py converts training sentences into positive and negative examples and extracts the features used for training (ngram, levensthein, etc). yields: training/link_train.csv with rows being samples: page, mention, link, feature1,...,featureN, label

Training the model:
- training a random forest to output link probability based on a set of features
  - generate_addlink_model.py trains the random forest. yields linkmodel.json

Evaluation:
- evluating on the backtesting dataset
  - analysis_eval-backtesting_prec-rec.py evaluates precision and recall on the testing sentences for different values of the threshold parameter. yields testing/backtest.eval.csv

Querying:
- convert pickle files to sqlite-tables
  - generate_sqlite_data.py converts the pickle files into sqlite-tables that can be used as dictionaries via sqlitedict for easier querying  (*.pkl --> *.sqlite)

## Querying the trained model

Once the model and all the utility files are computed (see "Training the model" below), they can be loaded and used to
build an API to add new links to a Wikipedia page automatically.

For this we can use the command line:

```bash
python query.py -p Berlin -id dewiki -m 3 -t 0.5
```
This will return all recommended links for a given page (-p) in a given wiki ID (-id) You can also specify the
threshold for the probability of the link (-t, default=0.5) and the maximum number of recommendations (-m, default=20).

## Analysis of the trained model

Scripts and notebooks for further evaluation of the models:

- summary statistics of the wikis (# articles, etc)
  - analysis_summary-stats.ipynb
- feature importance
  - analysis_model_feature-importance.ipynb
- summary of precision recall F1 from backtesting
  - analysis_eval_backtesting.ipynb
- number of articles with at least n recommendations from a random sample
  - analysis_number-of-recs-random.py get the data
  - analysis_number-of-recommendations-random_plot.ipynb plot the data
- subtask evaluation
  - entity linking
    - analysis_eval-backtesting_entitiy-linking.py
    - analysis_eval-entity-linking.ipynb
  - mention detection
    - analysis_eval-backtesting_mentions.py
    - analysis_eval-backtesting_mentions_plot.ipynb
