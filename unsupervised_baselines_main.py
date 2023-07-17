import argparse
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from baselines.tfidf import *
from baselines.bm25 import *
from baselines.textrank_bm25 import *
from baselines.rake_bm25 import *
import torch


def load_entity_list(path):
    entity_list = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            entity_list.append(line[1])
    return entity_list


def extract_target_data(data_path, description_path, ent_list):
    with open(description_path, 'r') as f:
        description_json = json.load(f)

    with open(data_path, 'r') as f:
        men_context_json = json.load(f)

    ent_des = {}
    men_doc_dict = {}
    labels = {}
    e2m = {}

    for ent in ent_list:
        ent_des.update({ent: description_json[ent]})
        labels.update({ent: []})
        men_doc_dict.update({ent: []})

        for name, doc_list in men_context_json[ent].items():
            e2m.update({ent:name})
            for doc_ctx in doc_list:
                labels[ent].append(doc_ctx["label"])
                doc_content = doc_ctx["left_context"] + name + doc_ctx["right_context"]
                men_doc_dict[ent].append(doc_content)

    return {'ent_des': ent_des, 'men_doc_dict': men_doc_dict, 'labels': labels, 'e2m': e2m}


def get_fold_data(data_path, description_path, ent_list, kfold, sample_fold):
    num = len(ent_list)
    fold_size = num // kfold
    train_ent = None
    val_ent = None
    test_ent = None
    for j in range(kfold):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        data_part = ent_list[idx]
        if j == sample_fold:
            test_ent = data_part
            
        elif train_ent is None:
            train_ent = data_part
        else:
            train_ent = train_ent + data_part
    val_ent = train_ent[:fold_size]
    train_ent = train_ent[fold_size:]

    val_dataset = extract_target_data(data_path, description_path, val_ent)
    test_dataset = extract_target_data(data_path, description_path, test_ent)

    return val_dataset, test_dataset


def calculate_f1(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > threshold).astype('int')

    TP = np.sum((pred == 1) * (labels == 1))
    TN = np.sum((pred == 0) * (labels == 0))
    FP = np.sum((pred == 1) * (labels == 0))
    FN = np.sum((pred == 0) * (labels == 1))
    acc = (TP + TN) * 1.0 / (TP + TN + FN + FP)
    if TP == 0:
        p = r = f1 = 0.0
    else:
        p = TP * 1.0 / (TP + FP)
        r = TP * 1.0 / (TP + FN)
        f1 = 2 * p * r / (p + r)
    return p, r, f1, acc


def test(model, dataset, threshold):
    ent_des, men_doc_dict, labels, e2m = dataset['ent_des'], dataset['men_doc_dict'], dataset['labels'], dataset['e2m']
    scores = []
    total_label = []
    for e, des in ent_des.items():
        target_label = labels[e]
        doc_list = men_doc_dict[e]
        m = e2m[e]
        sim_score = model.run(des, doc_list, target_label, m)
        if not isinstance(sim_score, list):
            sim_score = torch.sigmoid(sim_score)
            sim_score = sim_score.detach().tolist()
        scores.extend(sim_score)
        total_label.extend(target_label)

    p, r, f1, acc = calculate_f1(scores, total_label, threshold)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data path args
    parser.add_argument('--model', type=str, default='tf_idf')
    parser.add_argument('--entity_path', type=str, default='datasets/wiki300/target_entities.txt')
    parser.add_argument('--data_path', type=str, default='datasets/wiki300/TDD_dataset.json')
    parser.add_argument('--description_path', type=str, default='datasets/wiki300/entity_desc.json')
    parser.add_argument('--data_type', type=str, default='wiki')

    args = parser.parse_args()

    kfold = 5
    ent_list = load_entity_list(args.entity_path)
    result_path = args.result_path

    results = []
    thred_list = []
    ave_f1 = 0.0
    for i in range(kfold):
        if args.data_type == 'wiki':
            val_dataset, test_dataset = get_fold_data(args.data_path, args.description_path, ent_list, kfold, i)
        else:
            test_dataset = extract_target_data(args.data_path, args.description_path, ent_list)
            
        if args.model == 'tf_idf':
            # threshold---Wiki-100: 0.52  Wiki-200: 0.52 Wiki-300: 0.53  Web-Test: 0.54
            threshold = 0.54
            model = TFIDF()
            test_f1 = model.test_tf_idf(test_dataset, threshold)
        elif args.model == 'bm25':
            # threshold
            # Wiki100(fold 1-5): 0.55  0.95    0.86    0.02    0.3
            # Wiki200(fold 1-5): 0.96  0.01    0.99    0.94    0.94
            # Wiki300(fold 1-5): 0.77
            # Web_Test: 0.5
            threshold = 0.5
            model = BM25_Ranker()
            test_f1 = test(model, test_dataset, threshold)
        elif args.model == 'textrank_bm25':
            #threshold
            # Wiki100(fold 1-5): 0.03  0.01    0.01    0.1    0.01
            # Wiki200(fold 1-5): 0.01
            # Wiki300(fold 1-5): 0.53  0.5 0.1 0.07    0.01
            # Web_Test: 0.01
            threshold = 0.01
            model = TextRank_BM25()
            test_f1 = test(model, test_dataset, threshold)
        elif args.model == 'rake_bm25':
            # threshold
            # Wiki100(fold 1-5): 0.03  0.01    0.01    0.05    0.01
            # Wiki200(fold 1-5): 0.01
            # Wiki300(fold 1-5): 0.01  0.01    0.1 0.08        0.01
            # Web_Test(fold 1-5): 0.01
            threshold = 0.01
            model = Rake_BM25()
            test_f1 = test(model, test_dataset, threshold)

        results.append(test_f1)

    for i, r in enumerate(results):
        print(r)
        ave_f1 += r
    ave_f1 /= len(results)
    print('The average f1 of 5-fold is {}'.format(ave_f1))