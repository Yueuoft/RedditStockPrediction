from __future__ import annotations

from typing import Iterable

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentScorer:
    def __init__(self) -> None:
        self.analyzer = SentimentIntensityAnalyzer()

    def score_text(self, text: str) -> dict[str, float]:
        if not isinstance(text, str) or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
        return self.analyzer.polarity_scores(text)

    def score_frame(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        scores = out[text_col].apply(self.score_text).apply(pd.Series)
        scores.columns = [f"sent_{c}" for c in scores.columns]
        out = pd.concat([out, scores], axis=1)
        out["is_positive"] = (out["sent_compound"] > 0.05).astype(int)
        out["is_negative"] = (out["sent_compound"] < -0.05).astype(int)
        return out


def add_simple_hype_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    hype_terms = ["moon", "rocket", "squeeze", "diamond hands", "calls", "puts", "bullish", "bearish"]
    out = df.copy()
    lower = out[text_col].fillna("").str.lower()
    for term in hype_terms:
        col = f"kw_{term.replace(' ', '_')}"
        out[col] = lower.str.contains(term, regex=False).astype(int)
    out["hype_score"] = out[[c for c in out.columns if c.startswith("kw_")]].sum(axis=1)
    return out
