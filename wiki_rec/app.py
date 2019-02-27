import numpy as np

from fastapi import FastAPI
from starlette.background import BackgroundTask

from wiki_rec.ml.aggregation import make_average_aggregator
from wiki_rec.ml.candidates import make_faiss_candidate_generator
from wiki_rec.ml.ranking import make_cosine_ranker, make_model_ranker
from wiki_rec.ml.recommender import Recommender

from wiki_rec import model, db

app = FastAPI()
app.add_middleware(db.DatabaseInceptionMiddleware)

databases = {
    model.WikiEdition.EN: db.ArticlesDB(
        emb=np.load('emb-unsup-mean.npy'),
        idx2id=np.array([int(id.strip()) for id in open('id_map')]),
        titles=dict(line.strip().split('\t') for line in open('pages.tsv')
                    if len(line.strip().split('\t')) == 2)
    )
}

for edition, db in databases.items():
    db.register(edition.value)

rankers = {
    model.RankingEnum.COSINE: make_cosine_ranker(),
    # model.RankingEnum.DEEP_AVERAGE: make_model_ranker('/path/to/keras')
}

aggregators = {
    model.AggregatorEnum.AVERAGE: make_average_aggregator()
}

candidates = {
    model.WikiEdition.EN: make_faiss_candidate_generator(
        databases[model.WikiEdition.EN]
    )
}


@app.post('/api/{wiki_edition}/v1/recommend', response_model=model.RecommenderResponse)
async def recommend(wiki_edition: str,
                    request: model.RecommenderRequest):

    recommender = Recommender(
        aggregator=aggregators[request.aggregator],
        candidate_generator=candidates[model.WikiEdition(wiki_edition)],
        ranker=rankers[request.ranking],
        database=databases[model.WikiEdition(wiki_edition)]
    )

    task = BackgroundTask(recommender.recommend, request.user_history)
    await task()

    return {
        'recommendation': {
            'articles': recommender.recommendations[:request.top_n]
        }
    }
