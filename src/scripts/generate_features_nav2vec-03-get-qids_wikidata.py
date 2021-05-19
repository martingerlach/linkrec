import os, sys
import datetime
import calendar
import time
import string
import random
import pickle
import argparse
from pyspark.sql import functions as F, types as T, Window, SparkSession


"""
process webrequest table to get reading sessions
- returns filename where reading sessions are stored locally
    - ../data/<LANG>/<LANG>.reading-sessions

- USAGE:
PYSPARK_PYTHON=python3.7 PYSPARK_DRIVER_PYTHON=python3.7 spark2-submit --master yarn --executor-memory 8G --executor-cores 4 --driver-memory 2G  generate_features_nav2vec-01-get-sessions.py -l simple

- optional
    - t1, start-date (incusive)
    - t2, end-date (exclusive)

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-id",
        "-id",
        default=None,
        type=str,
        required=True,
        help="Wiki ID for which to get recommendations (e.g. enwiki)",
    )


    args = parser.parse_args()
    wiki_id = args.wiki_id


    ### start
    spark = (
        SparkSession.builder.master("yarn")
        .appName("qids")
        .enableHiveSupport()
        .getOrCreate()
    )

    w_wd = Window.partitionBy(F.col('wiki_db'),F.col('page_id')).orderBy(F.col('snapshot').desc())
    df_wd = (
        spark.read.table('wmf.wikidata_item_page_link')
        ## snapshot: this is a partition!
        .where(F.col('snapshot') >= '2020-07-01') ## resolve issues with non-mathcing wikidata-items
        ## only wikis (enwiki, ... not: wikisource)
        .where(F.col('wiki_db')==wiki_id)
        .withColumn('item_id_latest',F.first(F.col('item_id')).over(w_wd))
        .select(
            F.col('page_id').alias('pid'),
            F.col('item_id_latest').alias('qid')
        )
        .drop_duplicates()
    ).toPandas()

    ##save as dictionary
    dict_qids = df_wd.set_index("pid")["qid"].to_dict()
    output_path = "../../data/{0}/{0}.qids".format(wiki_id)
    with open(output_path + ".pkl", "wb") as handle:
        pickle.dump(
            dict_qids,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    main()
