from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np


def calculate_f1(scores, labels, thredhold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > thredhold).astype('int')

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


class TFIDF(object):

    def test_tf_idf(dataset, thredhold):
        ent_des, men_doc_dict, labels, e2m = dataset['ent_des'], dataset['men_doc_dict'], dataset['labels'], dataset[
            'e2m']
        scores = []
        total_label = []
        for e, des in ent_des.items():
            target_label = labels[e]
            doc_list = men_doc_dict[e]
            m = e2m[e]
            vectorizer = CountVectorizer(stop_words="english")
            tokenized_query = des.split(" ")
            tokenized_query = m.split(" ") + tokenized_query
            new_query = " ".join(tokenized_query)
            doc_list.insert(0, new_query)
            embeddings = vectorizer.fit_transform(doc_list)
            cos_sim = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
            doc_scores = torch.Tensor(cos_sim)
            sim_scores = torch.sigmoid(doc_scores).detach().tolist()
            scores.extend(sim_scores)
            total_label.extend(target_label)

        p, r, f1, acc = calculate_f1(scores, total_label, thredhold)
        return f1
