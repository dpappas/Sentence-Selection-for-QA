import  tensorflow as tf
import  sys, json
import  numpy as np
from    operator            import itemgetter

from    preprocess          import Word2Vec, BioASQ
from    ABCNN               import ABCNN
from    utils               import build_path
from    sklearn.externals   import joblib

def createJsonFile(dict, epoch):
    f = open('experiments/final_predictions_{}.{}.json'.format(epoch, mode), 'w')
    data = {'questions': []}
    for keys, candidates in dict.items():
        basic_info = {'body': keys[2], 'id': keys[1], 'snippets': []}
        counter = 0
        for answer, pred, old_q, old_a, os, oe, doc_id in sorted(candidates, key= itemgetter(1), reverse=True):
            if counter < 10:
                snips = {'document': "http://www.ncbi.nlm.nih.gov/pubmed/" + doc_id,
                         'text': old_a,
                         'offsetInBeginSection': int(os),
                         'offsetInEndSection': int(oe),
                         'beginSection': "abstract",
                         'endSection': "abstract"}
                counter += 1
                basic_info['snippets'].append(snips)
        data['questions'].append(basic_info)
    json.dump(data, f, indent=4)

def constructSentencefromIDs(ids, vocab):
    words = []
    for id in ids:
        if id != 0:
            words.append(wordFromID(id, vocab))
    sentence = " ".join(words)
    return sentence

def wordFromID(word_id, vocab):
    for key, value in vocab.items():
        try:
            if value == word_id:
                word = key
                return word
        except:
            print("Error...word not in Vocabulary!!!")

def test(mode, w, l2_reg, max_len, model_type, num_layers, data_type, classifier, word2vec, num_classes=2):
    if data_type == 'BioASQ':
        test_data = BioASQ(word2vec=word2vec, max_len=max_len)
    else:
        print("Wrong dataset...")
    ################################
    test_data.open_file(mode=mode)
    for epoch in range(epoch_from, epoch_to):
        tf.reset_default_graph()
        ################################
        model = ABCNN(s=max_len, w=w, l2_reg=l2_reg, model_type=model_type, d0=30, num_features=test_data.num_features, num_classes=num_classes, num_layers=num_layers)
        ################################
        model_path = build_path("./models/", data_type, model_type, num_layers)
        ################################
        MAPs, MRRs = [], []
        ################################
        print("=" * 50)
        print("test data size:", test_data.data_size)
        ################################
        test_data.reset_index()
        ################################
        with tf.Session() as sess:
            # max_epoch = 2    # Enter the best epoch during the evaluation of the validation set
            max_epoch = epoch
            saver = tf.train.Saver()
            saver.restore(sess, model_path + "-" + str(max_epoch))
            print(model_path + "-" + str(max_epoch), "restored.")
            ################################
            if classifier == "LR" or classifier == "SVM":
                clf_path = build_path("./models/", data_type, model_type, num_layers, "-" + str(max_epoch) + "-" + classifier + ".pkl")
                clf = joblib.load(clf_path)
                print(clf_path, "restored.")
            ################################
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('total_parameters: {}'.format(total_parameters))
            ################################
            QA_pairs = {}
            Bio_pairs = {}
            ################################
            s1s, s2s, labels, features = test_data.next_batch(batch_size=test_data.data_size)
            qids, old_qs, old_as, starts, ends, dids = test_data.getMoreInfo()
            ################################
            for i in range(test_data.data_size):
                pred, clf_input = sess.run(
                    [model.prediction, model.output_features],
                    feed_dict={
                        model.x1: np.expand_dims(s1s[i], axis=0),
                        model.x2: np.expand_dims(s2s[i], axis=0),
                        model.y: np.expand_dims(labels[i], axis=0),
                        model.features: np.expand_dims(features[i], axis=0)
                    }
                )
                ################################
                if classifier == "LR":
                    clf_pred = clf.predict_proba(clf_input)[:, 1]
                    pred = clf_pred
                elif classifier == "SVM":
                    clf_pred = clf.decision_function(clf_input)
                    pred = clf_pred
                ################################
                s1 = " ".join(test_data.s1s[i])
                s2 = " ".join(test_data.s2s[i])
                ################################
                if s1 in QA_pairs:
                    QA_pairs[s1].append((s2, labels[i], np.asscalar(pred)))
                    Bio_pairs[(s1, qids[i], old_qs[i])].append((s2, np.asscalar(pred), old_qs[i], old_as[i], starts[i], ends[i], dids[i]))
                else:
                    QA_pairs[s1] = [(s2, labels[i], np.asscalar(pred))]
                    Bio_pairs[(s1, qids[i], old_qs[i])] = [(s2, np.asscalar(pred), old_qs[i], old_as[i], starts[i], ends[i], dids[i])]
            ################################
            createJsonFile(Bio_pairs, epoch)
            # Calculate MAP and MRR
            MAP, MRR = 0, 0
            for s1 in QA_pairs.keys():
                p, AP = 0, 0
                MRR_check = False
                ################################
                QA_pairs[s1] = sorted(QA_pairs[s1], key=lambda x: x[-1], reverse=True)
                for idx, (s2, label, prob) in enumerate(QA_pairs[s1]):
                    if label == 1:
                        if not MRR_check:
                            MRR += 1 / (idx + 1)
                            MRR_check = True
                        p += 1
                        AP += p / (idx + 1)
                if p != 0:
                    AP /= p
                else:
                    AP = 0
                MAP += AP
            ################################
            num_questions = len(QA_pairs.keys())
            MAP /= num_questions
            MRR /= num_questions
            ################################
            MAPs.append(MAP)
            MRRs.append(MRR)

if __name__ == "__main__":
    # Parameters
    # --ws: window_size
    # --l2_reg: l2_reg modifier
    # --epoch: epoch
    # --max_len: max sentence length
    # --model_type: model type
    # --num_layers: number of convolution layers
    # --data_type: dataset name
    # --classifier: Final layout classifier(model, LR, SVM)
    ############################################################
    # default parameters
    mode        = sys.argv[1]
    epoch_from  = int(sys.argv[2])
    epoch_to    = int(sys.argv[3])
    params      = {
        "ws"            : 4,
        "l2_reg"        : 0.0004,
        "max_len"       : 40,
        "model_type"    : "BCNN",
        "num_layers"    : 2,
        "data_type"     : "BioASQ",
        "classifier"    : "LR",
        "word2vec"      : Word2Vec()
    }
    ############################################################
    # if len(sys.argv) > 1:
    #     for arg in sys.argv[1:]:
    #         k = arg.split("=")[0][2:]
    #         v = arg.split("=")[1]
    #         params[k] = v
    ############################################################
    test(
        mode        = mode,
        w           = int(params["ws"]),
        l2_reg      = float(params["l2_reg"]),
        max_len     = int(params["max_len"]),
        model_type  = params["model_type"],
        num_layers  = int(params["num_layers"]),
        data_type   = params["data_type"],
        classifier  = params["classifier"],
        word2vec    = params["word2vec"]
    )
    ############################################################

'''
cd ~/Sentence-Selection-for-QA/DL_Models/ABCNN/  
CUDA_VISIBLE_DEVICES=-1 python3.6 predictForTest.py dev  1 51
CUDA_VISIBLE_DEVICES=-1 python3.6 predictForTest.py test 50 51
# results are in dir : ./experiments/final_predictions_50.test.json

cp \
/home/dpappas/Sentence-Selection-for-QA/DL_Models/ABCNN/experiments/final_predictions_50.test.json \
/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_3/bert_bcnn.json

cp \
/home/dpappas/Sentence-Selection-for-QA/DL_Models/ABCNN/experiments/final_predictions_50.test.json \
/home/dpappas/bioasq_all/bioasq7/snippet_results/test_batch_4/bert_high_bcnn.json

'''

'''

for epoch in range(1, 51):
    print(
    'java -Xmx10G -cp "/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" evaluation.EvaluatorTask1b -phaseA -e 5 "/home/dpappas/bioasq_all/bioasq7_data/training7b.dev.json" "/home/dpappas/Sentence-Selection-for-QA/DL_Models/ABCNN/experiments/final_predictions_{}.dev.json" | grep "MAP snippets" | head -1'.format(epoch).strip()
    )

java \
-Xmx10G \
-cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA \
-e 5 \
"/home/dpappas/bioasq_all/bioasq7_data/training7b.dev.json" \
"/home/dpappas/Sentence-Selection-for-QA/DL_Models/ABCNN/experiments/final_predictions_50.dev.json" \
| grep "MAP snippets" | head -1


java \
-Xmx10G \
-cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA \
-e 5 \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/bert.json" \
| grep "MAP snippets" | head -1

java \
-Xmx10G \
-cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA \
-e 5 \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/snippet_results/test_batch_1/bert_high_bcnn.json" \
| grep "MAP snippets" | head -1

python3.6 "/home/dpappas/bioasq_all/eval/run_eval.py" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/bert.json" | grep map


java \
-Xmx10G \
-cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA \
-e 5 \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/snippet_results/test_batch_1/term_pacrr__bcnn.json" \
| grep "MAP snippets" | head -1

java \
-Xmx10G \
-cp \
"/home/dpappas/bioasq_all/dist/my_bioasq_eval_2.jar" \
evaluation.EvaluatorTask1b \
-phaseA \
-e 5 \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/data/test_batch_1/BioASQ-task7bPhaseB-testset1" \
"/home/dpappas/bioasq_all/bioasq7/bioasq7/document_results/test_batch_1/jpdrmm.json" \
| grep "MAP snippets" | head -1


'''
