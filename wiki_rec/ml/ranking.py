import numpy as np

from tensorflow.keras.models import load_model


def make_cosine_ranker():
    def rank(_user_repr, candidates, distances):
        return candidates[np.argsort(distances)]

    return rank


def make_model_ranker(keras_model):
    model = load_model(keras_model)

    def rank(user_repr, candidates, _distances):
        probs = model.predict({
            'user': np.tile(user_repr, (len(candidates), 1)),
            'target': candidates
        })
        return candidates[np.argsort(-probs)]

    return rank
