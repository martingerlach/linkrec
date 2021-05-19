import pickle
import argparse
import mwparserfromhell as mwph
import xgboost as xgb
import multiprocessing
import pandas as pd
import numpy as np
import operator
from utils import *
import json

list_wiki_id = ["arwiki","bnwiki","cswiki","viwiki","dewiki","ptwiki","simplewiki"]
nmax = 10000

for wiki_id in list_wiki_id:
    print(wiki_id)

    ## open dataset-dicts from pickle files
    anchors = pickle.load(open("../../data/{0}/{0}.anchors.pkl".format(wiki_id), "rb"))
    pageids = pickle.load(open("../../data/{0}/{0}.pageids.pkl".format(wiki_id), "rb"))
    redirects = pickle.load(open("../../data/{0}/{0}.redirects.pkl".format(wiki_id), "rb"))
    word2vec = pickle.load(
        open("../../data/{0}/{0}.w2vfiltered.pkl".format(wiki_id), "rb")
    )
    nav2vec = pickle.load(open("../../data/{0}/{0}.navfiltered.pkl".format(wiki_id), "rb"))

    ## load trained model
    ## use a fourth of the cpus, at most 8
    n_cpus_max = min([int(multiprocessing.cpu_count() / 4), 8])
    model = xgb.XGBClassifier(n_jobs=n_cpus_max)  # init model
    model.load_model("../../data/{0}/{0}.linkmodel.json".format(wiki_id))  # load data

    ## load the test-set
    test_set = []
    with open("../../data/{0}/testing/sentences_test.csv".format(wiki_id)) as fin:
        for line in fin:
            try:
                title, sent = line.split("\t")
                test_set.append((title, sent))
            except:
                continue

    output_path = "../../data/{0}/testing/{0}.backtest-eval_el_nmax-{1}.json".format(wiki_id,nmax)
    with open(output_path,'w', encoding='utf8') as fout:
        count_doc = 0
        for page, page_wikicode in test_set:
            try:
                input_code = page_wikicode
                ## get links from original wikitext (resolve redirects, and )
                inp_pairs = getLinks(input_code, redirects=redirects, pageids=pageids)

                ## if no links in main namespace, go to next item
                if len(inp_pairs) == 0:
                    continue

                dict_eval = {}
                dict_eval['page'] = page
                dict_eval_tmp = {}
                for anchor, title in inp_pairs.items():
                    title_predict, title_predict_pr = classify_links(page, anchor, anchors, word2vec, nav2vec, model, threshold=0.0)
                    dict_eval_tmp[anchor] = (title, title_predict, float(title_predict_pr))
                dict_eval["links"] = dict_eval_tmp
                count_doc += 1
                fout.write(json.dumps(dict_eval,ensure_ascii=False) + '\n')
            except:
                pass
            if count_doc == nmax:
                break





