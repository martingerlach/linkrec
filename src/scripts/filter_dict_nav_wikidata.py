import sys, os
import bz2, subprocess
import pickle
import glob
import numpy as np
import fasttext

if len(sys.argv) >= 2:
    wiki_id = sys.argv[1]
else:
    wiki_id = "enwiki"

## filter the embeddings and save as sqlite-tables
FILE_PAGEIDS = "../../data/{0}/{0}.pageids.pkl".format(wiki_id)
pageids = pickle.load(open(FILE_PAGEIDS, "rb"))

FILE_QIDS = "../../data/{0}/{0}.qids.pkl".format(wiki_id)
qids = pickle.load(open(FILE_QIDS, "rb"))

# embeddings from fasttext
navfile = "../../data/wikidata/wikidata.nav.bin".format(wiki_id)
nav2vec = fasttext.load_model(navfile)

N_kept = 0
nav2vec_filter = {}
for title in pageids.keys():
    pid = pageids[title]
    qid = qids.get(pid)
    if qid != None:
        vec = nav2vec.get_word_vector(qid)
        if np.abs(np.sum(vec))>0:
            nav2vec_filter[title] = np.array(vec)
            N_kept += 1

print(N_kept/len(pageids))
output_path = "../../data/{0}/{0}.navfiltered".format(wiki_id)
## dump as pickle
with open(output_path + ".pkl", "wb") as handle:
    pickle.dump(nav2vec_filter, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ## filter old files
# for FILENAME in glob.glob(navfile[:-4] + "*"):
#     if "filter" not in FILENAME:
#         os.system("rm -r %s" % FILENAME)
