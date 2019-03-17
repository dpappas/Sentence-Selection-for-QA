
import json, os, pickle, re, random
from pprint import pprint
from    nltk.tokenize import sent_tokenize

def get_gold_snips(quest_id, bioasq6_data):
    gold_snips = []
    if ('snippets' in bioasq6_data[quest_id]):
        for sn in bioasq6_data[quest_id]['snippets']:
            gold_snips.extend(sent_tokenize(sn['text']))
    return list(set(gold_snips))

def snip_is_relevant(one_sent, gold_snips):
    return any(
        [
            (one_sent.encode('ascii', 'ignore') in gold_snip.encode('ascii', 'ignore'))
            or
            (gold_snip.encode('ascii', 'ignore') in one_sent.encode('ascii', 'ignore'))
            for gold_snip in gold_snips
        ]
    )

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

dataloc = '/home/dpappas/bioasq_all/bioasq7_data/'

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

# train
# 55262a9787ecba3764000009        quest_text      sentence  tag
#
# dev
# 55242d512c8b63434a000006        quest_text      sentence  index_from   index_to     pubmed_id

for quer in train_data['queries']:
    rel_docs            = [rd for rd in quer['relevant_documents'] if(rd in train_docs)]
    query_id            = quer['query_id']
    query_text          = ' '.join(bioclean(quer['query_text']))
    snips               = [sn['text'] for sn in bioasq6_data[query_id]['snippets'] if(sn['document'].split('/')[-1].strip() in rel_docs)]
    good_snips          = [' '.join(bioclean(sn)) for sn in snips]
    all_rel, all_irel   = [], []
    for rel_doc in rel_docs:
        the_doc         = train_docs[rel_doc]
        sents           = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
        doc_rel_snips   = []
        doc_irel_snips  = []
        for sent in sents:
            sent = ' '.join(bioclean(sent))
            if(snip_is_relevant(sent , good_snips)):
                doc_rel_snips.append(sent)
            else:
                doc_irel_snips.append(sent)
        all_rel.extend(doc_rel_snips)
        all_irel.extend(doc_irel_snips)
    while(len(all_irel)<len(all_rel)):
        all_irel = all_irel + all_irel
    all_irel = random.sample(all_irel, len(all_rel))














