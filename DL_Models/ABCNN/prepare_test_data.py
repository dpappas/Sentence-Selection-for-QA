
import os, re, pickle, json
from tqdm import tqdm
from nltk import sent_tokenize
from pprint import pprint

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'", '').strip().lower()).split()

diri = './BioASQ_Corpus/'
if(not os.path.exists(diri)):
    os.makedirs(diri)

# f1 = '/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'
# f2 = '/home/dpappas/bioasq_all/bioasq7_data/test_batch_1/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'
# # docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/bert.json'
# docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/bert-high-conf-0.01.json'
# # docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/term-pacrr.json'

# f1 = '/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_2/bioasq7_bm25_top100/bioasq7_bm25_top100.test.pkl'
# f2 = '/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_2/bioasq7_bm25_top100/bioasq7_bm25_docset_top100.test.pkl'
# # docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_2/bert-high-conf-0.01.json'
# # docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_2/bert.json'
# docs_retrieved_path = '/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_2/term-pacrr.json'

import sys
batch               = sys.argv[1]
ddata               = '/home/dpappas/bioasq_all/bioasq7/data/test_batch_{}/bioasq7_bm25_top100/'.format(batch)
ddocs               = '/home/dpappas/bioasq_all/bioasq7/document_results/test_batch_{}/'.format(batch)
f1                  = os.path.join(ddata, 'bioasq7_bm25_top100.test.pkl')
f2                  = os.path.join(ddata, 'bioasq7_bm25_docset_top100.test.pkl')
docs_retrieved_path = os.path.join(ddocs, sys.argv[2])
# docs_retrieved_path = os.path.join(ddocs, 'bert-high-conf-0.01.json')
# docs_retrieved_path = os.path.join(ddocs, 'bert.json')
# docs_retrieved_path = os.path.join(ddocs, 'term-pacrr.json')
# docs_retrieved_path = os.path.join(ddocs, 'pdrmm.json')

with open(f1, 'rb') as f:
    test_data = pickle.load(f)

with open(f2, 'rb') as f:
    test_docs = pickle.load(f)

with open(docs_retrieved_path, 'r') as f:
    doc_res = json.load(f)
    doc_res = dict([(t['id'], t) for t in doc_res['questions']])

test_extracted_data = []
for quer in tqdm(test_data['queries']):
    query_id            = quer['query_id']
    retrieved_docs      = [rd.split('/')[-1] for rd in doc_res[query_id]['documents']]
    query_text          = quer['query_text']
    #################################
    for rel_doc in tqdm(retrieved_docs):
        the_doc         = test_docs[rel_doc]
        abs_sents       = sent_tokenize(the_doc['abstractText'])
        for sent in abs_sents:
            if(len(' '.join(bioclean(sent)).strip())!=0):
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

'''
cd ~/Sentence-Selection-for-QA/DL_Models/ABCNN/
python3.6 prepare_test_data.py
CUDA_VISIBLE_DEVICES=-1 python3.6 predictForTest.py test 50 51
mv \
/home/dpappas/Sentence-Selection-for-QA/DL_Models/ABCNN/experiments/final_predictions_50.test.json \
/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/term_pacrr_bcnn.json

scp \
/home/DATA/Biomedical/bioasq7/document_results/test_batch_3/ \
/home/DATA/Biomedical/bioasq7/document_results/test_batch_3/

python3.6 \
prepare_test_data.py 5 \
bert.json

python3.6 \
prepare_test_data.py 5 \
pdrmm.json

'''
