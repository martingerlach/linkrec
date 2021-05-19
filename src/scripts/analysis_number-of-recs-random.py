import xgboost as xgb
import sys,os
import json
import pickle
import numpy as np
import requests
import multiprocessing
from utils import normalise_title
from utils import getPageDict,process_page
import mwparserfromhell as mwph


list_wiki_id = ["arwiki","bnwiki","cswiki","viwiki","dewiki","ptwiki","simplewiki"]
list_threshold = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

N_random = 1000

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

    np.random.seed(2)
    list_pages = list(pageids.keys())
    np.random.shuffle(list_pages)

    N_count = 0
    N_loop = 0
    dict_page_wikitext = {}
    for page_title in list_pages:

        page_title = page_title.replace(' ','_')
        try:
            page_dict = getPageDict(page_title,wiki_id)
            wikitext = page_dict['wikitext']
            if len(wikitext)>1:
                dict_page_wikitext[page_title] = wikitext
                N_count +=1
        except:
            pass
        N_loop += 1
        if N_count == N_random:
            break

    # dict_page_lenrec = {}

    output_path = "../../data/{0}/testing/{0}.number-recs-random_{1}.json".format(wiki_id, N_random)
    with open(output_path,'w', encoding='utf8') as fout:
        for page_title, wikitext in dict_page_wikitext.items():
            dict_page = {}
            list_lenrec = []
            for threshold in list_threshold:
                added_links = process_page(
                    wikitext,
                    page_title,
                    anchors,
                    pageids,
                    redirects,
                    word2vec,
                    nav2vec,
                    model,
                    threshold=threshold,
                    return_wikitext = False)
                list_lenrec+=[len(added_links)]
            # dict_page_lenrec[page_title] = list_lenrec

            len_wikitext_char = len(wikitext)
            text = mwph.parse(wikitext).strip_code()
            len_text_char = len(text)

            dict_page['page'] = page_title
            dict_page['nrec'] = list_lenrec
            dict_page['len_wikitext_char'] = len_wikitext_char
            dict_page['len_text_char'] = len_text_char

            fout.write(json.dumps(dict_page,ensure_ascii=False) + '\n')

    # output_path = "../../data/{0}/testing/{0}.number-recs-random_{1}.json".format(wiki_id, N_random)
    # with open(output_path,'w', encoding='utf8') as fout:
    #     fout.write(json.dumps(dict_page_lenrec,ensure_ascii=False) + '\n')