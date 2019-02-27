import enum
import typing

from pydantic import BaseModel, ValidationError

from wiki_rec import db


class WikiEdition(enum.Enum):
    EN = 'en'
    UA = 'ua'


class RankingEnum(enum.Enum):
    COSINE = 'cosine'
    DEEP_AVERAGE = 'deep-average'
    DEEP_RANKING = 'deep-ranking'


class AggregatorEnum(enum.Enum):
    AVERAGE = 'average'
    RNN = 'rnn'


class Article(BaseModel):
    page_id: int = None
    title: str = None

    def __init__(self, *args, **kwargs):
        if kwargs.get('title') is None and kwargs.get('page_id') is None:
            raise ValidationError('At least one from article id or title must be specified')

        super().__init__(*args, **kwargs)

        if not self.title:
            current_db = db.get_current_db()
            self.title = current_db.titles[self.page_id]

        if not self.page_id:
            current_db = db.get_current_db()
            self.page_id = current_db.title2id[self.title]


class RecommenderRequest(BaseModel):
    user_id: typing.Optional[int]
    user_history: typing.Optional[typing.List[Article]]
    top_n: int = 50
    ranking: RankingEnum = RankingEnum.COSINE
    aggregator: AggregatorEnum = AggregatorEnum.AVERAGE


class Recommendation(BaseModel):
    articles: typing.List[Article]


class RecommenderResponse(BaseModel):
    recommendation: Recommendation


