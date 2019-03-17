
import json, os, pickle

dataloc             = '/home/dpappas/bioasq_all/bioasq7_data/'

with open(os.path.join(dataloc, 'trainining7b.json'), 'r') as f:
    bioasq6_data = json.load(f)
    bioasq6_data = dict((q['id'], q) for q in bioasq6_data['questions'])

with open(os.path.join(dataloc, 'bioasq7_bm25_top100.dev.pkl'), 'rb') as f:
    dev_data = pickle.load(f)

with open(os.path.join(dataloc, 'bioasq7_bm25_docset_top100.dev.pkl'), 'rb') as f:
    dev_docs = pickle.load(f)

with open(os.path.join(dataloc, 'bioasq7_bm25_top100.train.pkl'), 'rb') as f:
    train_data = pickle.load(f)

with open(os.path.join(dataloc, 'bioasq7_bm25_docset_top100.train.pkl'), 'rb') as f:
    train_docs = pickle.load(f)













