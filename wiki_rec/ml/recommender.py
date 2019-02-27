import typing
#from functools import lru_cache

from wiki_rec.model import Article
from wiki_rec import db


#@lru_cache
class Recommender:
    def __init__(self,
                 aggregator: typing.Callable,
                 candidate_generator: typing.Callable,
                 ranker: typing.Callable,
                 database: db.ArticlesDB):
        self.candidate_generator = candidate_generator
        self.ranker = ranker
        self.aggregator = aggregator
        self.database = database

        self.recommendations = []

    def recommend(self, user_history: typing.List[Article]):
        articles_idx = [self.database.id2idx[article.page_id] for article in user_history]
        articles_emb = [self.database.emb[idx] for idx in articles_idx]

        user_repr = self.aggregator(articles_emb)

        candidates, distances = self.candidate_generator(user_repr)

        candidates_sorted = self.ranker(user_repr, candidates, distances)
        with db.database(self.database):
            self.recommendations = [Article(page_id=self.database.idx2id[idx])
                                    for idx in candidates_sorted]

