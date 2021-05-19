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

    output_path = "../../data/{0}/testing/{0}.backtest-eval_mentions_nmax-{1}.json".format(wiki_id,nmax)
    with open(output_path,'w', encoding='utf8') as fout:
        count_doc = 0
        for page, page_wikicode in test_set:
            # try:
            input_code = page_wikicode
            ## get links from original wikitext (resolve redirects, and )
            inp_pairs = getLinks(input_code, redirects=redirects, pageids=pageids)

            ## if no links in main namespace, go to next item
            if len(inp_pairs) == 0:
                continue
            input_code_nolinks = mwph.parse(page_wikicode).strip_code()

            linked_mentions=set(normalise_anchor(page))
            linked_links=set(normalise_title(page))
            tested_mentions = set()

            page_wikicode_init = str(input_code_nolinks)  # save the initial state
            page_wikicode_text_nodes = mwparserfromhell.parse(input_code_nolinks).filter_text(recursive=False)



            list_topcand = []
            for node in page_wikicode_text_nodes:
                for gram in ngram_iterator(node, 10, 1):
                    mention = gram.lower()
                    mention_original = gram
        #             print(mention)
                    # if the mention exist in the DB
                    # it was not previously linked (or part of a link)
                    # none of its candidate links is already used
                    # it was not tested before (for efficiency)
                    if (
                        mention in anchors
                        and not any(mention in s for s in linked_mentions)
                        and not bool(set(anchors[mention].keys()) & linked_links)
                        and mention not in tested_mentions
                    ):
                        tested_mentions.add(mention)

                        anchor = mention
                        cand_prediction = {}
                        # Work with the 10 most frequent candidates
                        limited_cands = anchors[anchor]
                        if len(limited_cands) > 10:
                            limited_cands = dict(
                                sorted(anchors[anchor].items(), key=operator.itemgetter(1), reverse=True)[:10]
                            )
                        for cand in limited_cands:
                            cand_feats = get_feature_set(page, anchor, cand, anchors, word2vec, nav2vec)
                            # compute the model probability
                            cand_prediction[cand] = model.predict_proba(
                                np.array(cand_feats).reshape((1, -1))
                            )[0, 1]
                        # Compute the top candidate
                        top_candidate = max(cand_prediction.items(), key=operator.itemgetter(1))
                        list_topcand += [(anchor, float(top_candidate[1]))]

            dict_eval = {}
            dict_eval["page"] = page
            dict_eval["links"] = inp_pairs
            dict_eval["mentions"] = list_topcand
            count_doc += 1
            fout.write(json.dumps(dict_eval,ensure_ascii=False) + '\n')
            # except:
            #     pass
            if count_doc == nmax:
                break





