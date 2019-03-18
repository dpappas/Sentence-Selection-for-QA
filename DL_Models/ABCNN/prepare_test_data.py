
import os, re, pickle
from tqdm import tqdm
from nltk import sent_tokenize

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

diri = './BioASQ_Corpus/'
if(not os.path.exists(diri)):
    os.makedirs(diri)

with open('/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl', 'rb') as f:
    test_data = pickle.load(f)

with open('/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl', 'rb') as f:
    test_docs = pickle.load(f)

test_extracted_data = []
for quer in tqdm(test_data['queries']):
    retrieved_docs      = [rd.split('/')[-1] for rd in  if(rd in test_docs)]
    query_id            = quer['query_id']
    query_text          = quer['query_text']
    #################################
    for rel_doc in tqdm(retrieved_docs):
        the_doc         = test_docs[rel_doc]
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


with open(os.path.join(diri, 'BioASQ-test.txt'), 'w') as f:
    for d in test_extracted_data:
        f.write('\t'.join(d).replace('\n', ' ') + '\n')
    f.close()
