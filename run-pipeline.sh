#!/bin/bash

set -ex

# on stat-machine you might have to "kinit" first

WIKI_ID=${WIKI_ID:-simplewiki}

## go to scripts directory
cd src/scripts/

# # create folder for data
echo "CREATING FOLDERS for data in ../../data/${WIKI_ID}"

mkdir ../../data/$WIKI_ID
mkdir ../../data/$WIKI_ID/training
mkdir ../../data/$WIKI_ID/testing


# echo 'GETTING THE ANCHOR DICTIONARY'
# # for the anchor dictionary we use the conda-environment on stats
source /usr/lib/anaconda-wmf/bin/activate
PYSPARK_PYTHON=/usr/lib/anaconda-wmf/bin/python3.7 PYSPARK_DRIVER_PYTHON=/usr/lib/anaconda-wmf/bin/python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G --conf spark.dynamicAllocation.maxExecutors=128 generate_anchor_dictionary_spark.py $WIKI_ID

# get wikidata-properties to filter, e.g., dismabiguation pages as links
PYSPARK_PYTHON=/usr/lib/anaconda-wmf/bin/python3.7 PYSPARK_DRIVER_PYTHON=/usr/lib/anaconda-wmf/bin/python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G --conf spark.dynamicAllocation.maxExecutors=128 generate_wdproperties_spark.py $WIKI_ID
python filter_dict_anchor.py $WIKI_ID
conda deactivate


# activate the custom virtual environment
source ../../venv/bin/activate
# alternatively, one can get the anchor-dictionary by processing the xml-dumps
# note that this does not filter by link-probability
# python generate_anchor_dictionary.py $WIKI_ID


## get wikipedia2vec-mebddingq
echo 'RUNNING wikipedia2vec on dump'
ionice wikipedia2vec train \
  --min-entity-count=0 \
  --dim-size 50 \
  --pool-size 8 \
  "/mnt/data/xmldatadumps/public/${WIKI_ID}/latest/${WIKI_ID}-latest-pages-articles.xml.bz2" \
  "../../data/${WIKI_ID}/${WIKI_ID}.w2v.bin"

python filter_dict_w2v.py $WIKI_ID


# get navigation features (remove the reading sessions and only keep the model)
echo 'RUNNING nav2vec'
## this approach trains an embedding on reading sessions in each wiki separately
# PYSPARK_PYTHON=python3.7 PYSPARK_DRIVER_PYTHON=python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G --conf spark.dynamicAllocation.maxExecutors=128 generate_features_nav2vec-01-get-sessions.py -id $WIKI_ID
# python generate_features_nav2vec-02-train-w2v.py -id $WIKI_ID -rfin True -w 16
# python filter_dict_nav.py $WIKI_ID

## this approach trains an embedding on readins sessions in all wikipedias combined by mapping to qids
## we then get the embedding of each wiki by mapping back to pageids and filtering pages that exist in the wikis
## one needs to train the embedding only once (see commented commands)
# PYSPARK_PYTHON=python3.7 PYSPARK_DRIVER_PYTHON=python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G --conf spark.dynamicAllocation.maxExecutors=128 generate_features_nav2vec-01-get-sessions_wikidata.py
# python generate_features_nav2vec-02-train-w2v.py -id wikidata -rfin False -w 16
# projecting embedding of all wikis into wiki
PYSPARK_PYTHON=python3.7 PYSPARK_DRIVER_PYTHON=python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G --conf spark.dynamicAllocation.maxExecutors=128 generate_features_nav2vec-03-get-qids_wikidata.py -id $WIKI_ID
python filter_dict_nav_wikidata.py $WIKI_ID


# # generate backtesting data
echo 'GENERATING BACKTESTIN DATA'
python generate_backtesting_data.py $WIKI_ID

# turn into features
echo 'GENERATING FEATURES'
python generate_training_data.py $WIKI_ID

#  train model
echo 'TRAINING THE MODEL'
python generate_addlink_model.py $WIKI_ID

# ## perform automatic backtesting
echo 'RUNNING BACKTESTING EVALUATION'
python analysis_eval-backtesting_prec-rec.py -id $WIKI_ID -nmax 10000

# # converting data to shelve format
echo 'CONVERTING DATA TO SQLITE FORMAT'
python generate_sqlite_data.py $WIKI_ID

deactivate