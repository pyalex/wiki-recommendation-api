import numpy as np
import typing
import threading
import contextlib
import functools

from dataclasses import dataclass
from starlette.requests import Request
from starlette.types import Scope, ASGIInstance, Receive, Send, ASGIApp

_registry = {}
_current_db = threading.local()


@dataclass
class ArticlesDB:
    emb: np.ndarray
    idx2id: np.ndarray
    titles: typing.Dict[int, str]

    @property
    def id2idx(self):
        if not hasattr(self, '_id2idx'):
            self._id2idx = {id: idx for idx, id in enumerate(self.idx2id)}

        return self._id2idx
    
    @property
    def title2id(self):
        if not hasattr(self, '_title2id'):
            self._title2id = {title: id for id, title in self.titles.items()}
            
        return self._title2id

    def register(self, key):
        _registry[key] = self


def set_current_db(db: ArticlesDB):
    setattr(_current_db, '_current_db', db)
    
    
def get_current_db() -> ArticlesDB:
    return getattr(_current_db, '_current_db', None)


@contextlib.contextmanager
def database(db: ArticlesDB):
    set_current_db(db)
    yield
    set_current_db(None)


class DatabaseInceptionMiddleware:
    def __init__(
        self, app: ASGIApp, handler: typing.Callable = None, debug: bool = False
    ) -> None:
        self.app = app
        self.handler = handler
        self.debug = debug

    def __call__(self, scope: Scope) -> ASGIInstance:
        if scope["type"] != "http":
            return self.app(scope)
        return functools.partial(self.asgi, scope=scope)

    async def asgi(self, receive: Receive, send: Send, scope: Scope) -> None:
        asgi = self.app(scope)

        # run the router
        self.app.app(scope)
        params = scope.get('path_params', {})
        if not params.get('wiki_edition'):
            await asgi(receive, send)

        with database(_registry[params['wiki_edition']]):
            await asgi(receive, send)

