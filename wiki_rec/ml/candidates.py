import faiss
import numpy as np

from wiki_rec.db import ArticlesDB


def make_faiss_candidate_generator(database: ArticlesDB, n=2):
    index = faiss.IndexHNSWFlat(database.emb.shape[1], n)
    #known_idxs = np.array(idx2id.keys())

    index.train(database.emb.astype(np.float32))
    index.add(database.emb.astype(np.float32))

    def generator(user_repr, limit=500):
        query = np.expand_dims(user_repr, 0)
        distances, idxs = index.search(query.astype(np.float32), limit)
        #ids = [idx2id[idx] for idx in idxs[np.isin(idxs, known_idxs)]]
        #distances = distances[np.isin(idxs, known_idxs)]
        return idxs.ravel(), distances.ravel()

    return generator
