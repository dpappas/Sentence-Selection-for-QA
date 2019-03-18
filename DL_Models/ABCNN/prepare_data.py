
import json, os, pickle, re, random
from pprint import pprint
from    nltk.tokenize import sent_tokenize
from tqdm import tqdm

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

with open('/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl', 'rb') as f:
    test_data = pickle.load(f)

with open('/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl', 'rb') as f:
    test_docs = pickle.load(f)

# train
# 55262a9787ecba3764000009        quest_text      sentence  tag
#
# dev
# 55242d512c8b63434a000006        quest_text      sentence  index_from   index_to     pubmed_id

train_extracted_data = []
for quer in tqdm(train_data['queries']):
    rel_docs            = [rd for rd in quer['relevant_documents'] if(rd in train_docs)]
    query_id            = quer['query_id']
    query_text          = ' '.join(bioclean(quer['query_text']))
    snips               = [sn['text'] for sn in bioasq6_data[query_id]['snippets'] if(sn['document'].split('/')[-1].strip() in rel_docs)]
    good_snips          = [' '.join(bioclean(sn)) for sn in snips]
    all_rel, all_irel   = [], []
    for rel_doc in tqdm(rel_docs):
        the_doc         = train_docs[rel_doc]
        # sents           = sent_tokenize(the_doc['title']) + sent_tokenize(the_doc['abstractText'])
        sents           = sent_tokenize(the_doc['abstractText'])
        doc_rel_snips   = []
        doc_irel_snips  = []
        for sent in sents:
            sent = ' '.join(bioclean(sent)).strip()
            if(len(sent)>0):
                if(snip_is_relevant(sent , good_snips)):
                    doc_rel_snips.append(sent)
                else:
                    doc_irel_snips.append(sent)
        all_rel.extend(doc_rel_snips)
        all_irel.extend(doc_irel_snips)
    if(len(all_irel)==0):
        continue
    while(len(all_irel)<len(all_rel)):
        all_irel = all_irel + all_irel
    all_irel = random.sample(all_irel, len(all_rel))
    #################################
    for item in tqdm(zip(all_rel, all_irel)):
        train_extracted_data.append([query_id, query_text, item[0], '1'])
        train_extracted_data.append([query_id, query_text, item[1], '0'])

print(len(train_extracted_data))

############################################################

dev_extracted_data = []
for quer in tqdm(dev_data['queries']):
    rel_docs            = [rd for rd in quer['relevant_documents'] if(rd in dev_docs)]
    query_id            = quer['query_id']
    query_text          = quer['query_text']
    #################################
    for rel_doc in tqdm(rel_docs):
        the_doc         = dev_docs[rel_doc]
        # tit_sents       = sent_tokenize(the_doc['title'])
        # for sent in tit_sents:
        #     if(len(' '.join(bioclean(query_text)).strip())!=0):
        #         dev_extracted_data.append(
        #             [
        #                 query_id, ' '.join(bioclean(query_text)), ' '.join(bioclean(sent)), query_text, sent,
        #                 str(the_doc['title'].index(sent)), str(the_doc['title'].index(sent)+len(sent)), rel_doc
        #             ]
        #         )
        abs_sents = sent_tokenize(the_doc['abstractText'])
        for sent in abs_sents:
            if(len(' '.join(bioclean(query_text)).strip())!=0):
                dev_extracted_data.append(
                    [
                        query_id, ' '.join(bioclean(query_text)), ' '.join(bioclean(sent)), query_text, sent,
                        str(the_doc['abstractText'].index(sent)), str(the_doc['abstractText'].index(sent)+len(sent)), rel_doc
                    ]
                )

print(len(dev_extracted_data))

############################################################

test_extracted_data = []
for quer in tqdm(test_data['queries']):
    rel_docs            = [rd for rd in quer['relevant_documents'] if(rd in test_docs)]
    query_id            = quer['query_id']
    query_text          = quer['query_text']
    #################################
    for rel_doc in tqdm(rel_docs):
        the_doc         = test_docs[rel_doc]
        # tit_sents       = sent_tokenize(the_doc['title'])
        # for sent in tit_sents:
        #     if(len(' '.join(bioclean(query_text)).strip())!=0):
        #         dev_extracted_data.append(
        #             [
        #                 query_id, ' '.join(bioclean(query_text)), ' '.join(bioclean(sent)), query_text, sent,
        #                 str(the_doc['title'].index(sent)), str(the_doc['title'].index(sent)+len(sent)), rel_doc
        #             ]
        #         )
        abs_sents       = sent_tokenize(the_doc['abstractText'])
        for sent in abs_sents:
            if(len(' '.join(bioclean(query_text)).strip())!=0):
                test_extracted_data.append(
                    [
                        query_id, ' '.join(bioclean(query_text)), ' '.join(bioclean(sent)), query_text, sent,
                        str(the_doc['abstractText'].index(sent)), str(the_doc['abstractText'].index(sent)+len(sent)), rel_doc
                    ]
                )

print(len(test_extracted_data))

############################################################

diri = './BioASQ_Corpus/'
if(not os.path.exists(diri)):
    os.makedirs(diri)

with open(os.path.join(diri, 'BioASQ-train.txt'), 'w') as f:
    for d in train_extracted_data:
        f.write('\t'.join(d).replace('\n', ' ') + '\n')
    f.close()

with open(os.path.join(diri, 'BioASQ-dev.txt'), 'w') as f:
    for d in dev_extracted_data:
        f.write('\t'.join(d).replace('\n', ' ') + '\n')
    f.close()

with open(os.path.join(diri, 'BioASQ-test.txt'), 'w') as f:
    for d in test_extracted_data:
        f.write('\t'.join(d).replace('\n', ' ') + '\n')
    f.close()

############################################################

'''
head -10 ./BioASQ_Corpus7/BioASQ-train.txt
'''

'''
train
qid \t clean_quest \t clean_sent \t tag

dev
qid \t clean_quest \t clean_sent \t quest \t sent \t offset_begin \t offset_end \t pubmed_id
'''





