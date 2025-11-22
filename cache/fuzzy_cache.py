from typing import Dict, Optional

import pandas as pd
from fuzzywuzzy import fuzz

from cache.wrapper import CacheResult, CacheResults


class FuzzyCache:
    def __init__(self):
        self.store = []

    def hydrate_from_df(
        self,
        df: pd.DataFrame,
        *,
        q_col: str = "question",
        a_col: str = "answer",
        clear: bool = True,
    ):
        if clear:
            self.store = []
        idx = 0
        for row in df[[q_col, a_col]].itertuples(index=False, name=None):
            q, a = row
            self.store.append([q, a])
            idx += 1

    def check_many(self, queries, distance_threshold: Optional[float] = None):
        distance_threshold = 1 if distance_threshold is None else distance_threshold

        results = []
        for query in queries:
            max_ratio = 0
            matched = None
            for q, a in self.store:
                ratio = fuzz.ratio(q, query)
                if ratio > max_ratio:
                    max_ratio = ratio
                    matched = [q, a]

            matched_query, answer = matched
            max_ratio = 1 - (max_ratio / 100.0)
            matches = [
                CacheResult(
                    prompt=matched_query,
                    response=answer,
                    vector_distance=max_ratio,
                    # not really cosine similarity, but we have to match the interface
                    cosine_similarity=1 - max_ratio,
                )
            ]
            if max_ratio > distance_threshold:
                matches = []
            results.append(CacheResults(query=query, matches=matches))
        return results
