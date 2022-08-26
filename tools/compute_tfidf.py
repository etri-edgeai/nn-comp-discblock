""" TFIDF computation utility """

import math
import pickle

def compute(
    ntokens,
    docs,
    data_name,
    mode="tfidf",
    pad_token=None,
    alpha=0.5,
    beta=0.25,
    base=0,
    t_ntokens=None,
    t_docs=None):
    """ TFIDF score computation """

    score = _compute(ntokens, docs, mode=mode, pad_token=pad_token, alpha=alpha, beta=beta, base=base)
    if t_ntokens is not None:
        t_score = _compute(t_ntokens, t_docs, mode=mode, pad_token=pad_token, alpha=alpha, beta=beta, base=base)
        score = (score, t_score)

    if mode == "tfidf":
        with open("tf_score_%s_%.3f_%.3f.pkl" % (data_name, alpha, beta), "wb") as f:
            pickle.dump(score, f)

    elif mode == "frequency":
        if base <= 1:
            with open("f_score_%s.pkl" % data_name, "wb") as f:
                pickle.dump(score, f)
        else:
            with open("f_score_%s_%d.pkl" % (data_name, base), "wb") as f:
                pickle.dump(score, f)


def _compute(ntokens, docs, mode="tfidf", pad_token=None, alpha=0.5, beta=0.25, base=1):
    """ TFIDF score computation imple. """

    tf = { t:{} for t in range(ntokens) }
    idf = { t:1 for t in range(ntokens) }
    max_tf = {}
    
    temp = []
    cnt = 0
    for batch in docs: # batch-docs
        for doc in batch:
            for w in doc:
                if w == pad_token:
                    continue
                if cnt not in tf[w]:
                    tf[w][cnt] = 0
                    idf[w] += 1
                tf[w][cnt] += 1
            if pad_token is not None:
                tf[pad_token][cnt] = 0.0
            max_tf[cnt] = max([tf[w][cnt] for w in doc])
            cnt += 1

    tf_ = {}
    print(alpha * cnt, ntokens)
    for t in range(ntokens):
        tf_sum_ = alpha * cnt
        for d in range(cnt):
            tf__ = tf[t][d] if d in tf[t] else 0
            tf_sum_ += beta * float(tf__) / max_tf[d]
        tf_[t] = tf_sum_ / cnt
        idf[t] = max(math.log(float(cnt) / (idf[t]+1)), 0.0) + 1

    if mode == "tfidf":
        score = [(tf_[t] * idf[t]) + 1/cnt for t in range(ntokens)]
    elif mode == "frequency":
        f = {}
        for t in range(ntokens):
            sum_ = 0
            for d in range(cnt):
                if d in tf[t]:
                    sum_ += tf[t][d]
            f[t] = sum_
        score = [(f[t])+base for t in range(ntokens)]
    if pad_token is not None and mode == "tfidf":
        score[pad_token] = 1/cnt
    elif pad_token is not None:
        score[pad_token] = 0
    print(score[0])
    print(score[-1])
    return score
