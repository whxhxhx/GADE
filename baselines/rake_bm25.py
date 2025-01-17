# import spacy
import torch
from torch import nn
import math
from collections import defaultdict, Counter
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
# import pytextrank
from rake_nltk import Rake

k1 = 1.2
b = 0.75
doc_len=800
ranker_vocab = dict((x[1], x[0]) for x in enumerate([line.strip() for line in open("embed_folder/GoogleNews-vectors-negative300_vocab.txt")]))
def get_tf_idf_dict(docments, doc_len=1000):
    tf_dict = {}
    df_dict = {}
    doc_tensors = []
    stop_words = list(stopwords.words('english'))
    limit = 6000
    idx = 0
    for doc in docments:
        doc = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", doc)
        doc = re.sub(r"\s+", " ", doc)
        doc = word_tokenize(doc)
        doc = [x for x in doc if x not in stop_words and len(x) > 2 and x.isalpha()]
        doc = [ranker_vocab[x] for x in doc if x in ranker_vocab]
        if len(doc) > doc_len:
            doc = doc[:doc_len]
        else:
            doc.extend([-1] * (doc_len - len(doc)))
        tf_dict[idx] = dict(Counter(doc))
        for tok in set(doc):
            df_dict[tok] = df_dict.get(tok, 0.0) + 1
        doc_tensors.append(torch.LongTensor([doc]))
        idx += 1
    return doc_tensors, tf_dict, df_dict


r = Rake(max_length=3)
max_keyw = 50


class Rake_BM25(object):

    def rum_rake_bm25(self, query, doc_tensors, tf_dict, ranker, mention):
        stop_words = list(stopwords.words('english'))
        query = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", query)
        query = re.sub(r"\s+", " ", query)
        query = word_tokenize(query)
        query = [x for x in query if x not in stop_words and len(x) > 2 and x.isalpha()]
        query = ' '.join(query)
        # mention = mention.split(' ')
        mention = word_tokenize(mention)
        # mention.remove('(')
        # mention.remove(')')
        r.extract_keywords_from_text(query)
        m_len = len(mention)
        phrases = r.get_ranked_phrases()[:max_keyw-m_len]
        phrases = mention + phrases
        if len(phrases) < max_keyw:
            phrases.extend(['-1'] * (max_keyw - len(phrases)))
        
        query = [ranker_vocab.get(term, 0) for term in phrases]
        score_list = []
        df = [ranker.df_dict.get(term, 0) for term in query]
        idf = torch.tensor([math.log((ranker.doc_num - termdf + 0.5) / (termdf + 0.5)) for termdf in df])
        for num, doc in enumerate(doc_tensors):
            qtf = torch.FloatTensor([tf_dict[num].get(term, 0) for term in query])
            score_list.append(ranker(qtf, idf, doc))

        return score_list

    def run(self, query, documents, labels, mention):
        doc_tensors, tf_dict, df_dict = get_tf_idf_dict(documents, doc_len=doc_len)
        ranker = _bm25(df_dict=df_dict, k1=k1, b=b, doc_num=len(doc_tensors))
        score_list = self.rum_rake_bm25(query, doc_tensors, tf_dict, ranker, mention)
        score_list = torch.stack(score_list)
        
        return score_list


class _bm25(nn.Module):
    def __init__(self, df_dict, k1, b, doc_num=0, doc_len=0):
        super(_bm25, self).__init__()
        self.k1 = torch.tensor(k1)
        self.b = torch.tensor(b)
        self.doc_num, self.doc_len = doc_num, doc_len
        self.df_dict = {tok: torch.tensor(df) for tok, df in df_dict.items()} if df_dict != None else None

    def forward(self, qtf, idf, d, avg_dl, without_idf = False):
        num = qtf * (self.k1 + 1)
        denom = qtf + self.k1 * (1 - self.b + self.b * len(d) / avg_dl)

        if not without_idf:
            scores = idf * (num / denom)
        else:
            scores = num /denom

        return torch.sum(scores)
