import numpy as np


def make_average_aggregator():
    def agg(article_embs):
        return np.mean(article_embs, 0)

    return agg