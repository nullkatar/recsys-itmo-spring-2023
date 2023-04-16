import numpy as np

from .random import Random
from .recommender import Recommender
import random
import joblib


def norm(x):
  return (x - x.mean(axis=0)) / x.std(axis=0)

class HOMEWORK(Recommender):
    """
    Recommend tracks closest to the previous one.
    Fall back to the random recommender if no
    recommendations found for the track.
    """

    def __init__(self, tracks_redis, recommendations_redis, catalog, emb_path, context_path, track_path):    
        self.recommendations_redis = recommendations_redis
        self.tracks_redis = tracks_redis
        self.fallback = Random(tracks_redis)
        self.catalog = catalog
        self.user_embs = np.load(emb_path)
        self.context_embs = np.load(context_path)
        self.track_embs = np.load(track_path)

    def recommend_next(self, user: int, prev_track: int, prev_track_time: float) -> int:
        previous_track = self.tracks_redis.get(prev_track)
        if previous_track is None:
            return self.fallback.recommend_next(user, prev_track, prev_track_time)

        previous_track = self.catalog.from_bytes(previous_track).track
        embedding = self.user_embs[user] + self.context_embs[previous_track]
        recommendations = np.argpartition(-np.dot(self.track_embs, embedding), 100)[:100].tolist()

        shuffled = list(recommendations)
        random.shuffle(shuffled)
        return shuffled[0]
