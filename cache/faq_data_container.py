import numpy as np
import pandas as pd


class FAQDataContainer:
    def __init__(self):
        self.faq_df = pd.read_csv("data/faq_seed.csv")
        self.test_df = pd.read_csv("data/test_dataset.csv")

        print(f"Loaded {len(self.faq_df)} FAQ entries")
        print(f"Loaded {len(self.test_df)} test queries")

    def _resolve_question(self, q):
        matches = self.test_df[self.test_df["question"] == q]
        assert len(matches) == 1, "each hit should be matched to a src item"
        src_question_id = matches["src_question_id"].iloc[0]
        should_hit = matches["cache_hit"].iloc[0]
        expected_hit = self.faq_df.iloc[src_question_id]["question"]
        expected_hit = expected_hit if should_hit else None
        return expected_hit

    def label_cache_hits(self, cache_results):
        results = []
        test_qs = self.test_df["question"].tolist()
        for res in cache_results:
            expected_hit = self._resolve_question(res.query)
            actual_hit = None if len(res.matches) == 0 else res.matches[0].prompt

            if actual_hit is not None and actual_hit in test_qs:
                actual_hit = self._resolve_question(actual_hit)

            results.append(expected_hit == actual_hit)

        return np.array(results)
