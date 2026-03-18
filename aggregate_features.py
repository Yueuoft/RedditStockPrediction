from __future__ import annotations

import pandas as pd


def ensure_datetime_date(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out["date"] = out[ts_col].dt.tz_convert(None).dt.floor("D")
    return out


def aggregate_reddit_features(mentions_df: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "score", "author", "sent_compound", "is_positive", "is_negative", "hype_score"}
    missing = required.difference(mentions_df.columns)
    if missing:
        raise ValueError(f"mentions_df is missing required columns: {sorted(missing)}")

    out = mentions_df.copy()
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0)
    out["score_pos"] = out["score"].clip(lower=0)
    out["weighted_sentiment"] = out["sent_compound"] * (1.0 + out["score_pos"])

    grouped = out.groupby(["date", "ticker"], as_index=False).agg(
        mentions=("ticker", "size"),
        unique_authors=("author", pd.Series.nunique),
        avg_score=("score", "mean"),
        total_score=("score", "sum"),
        avg_sentiment=("sent_compound", "mean"),
        frac_positive=("is_positive", "mean"),
        frac_negative=("is_negative", "mean"),
        avg_hype_score=("hype_score", "mean"),
        weighted_sentiment=("weighted_sentiment", "sum"),
    )

    grouped = grouped.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped["mentions_ma_7"] = grouped.groupby("ticker")["mentions"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    grouped["mentions_std_7"] = grouped.groupby("ticker")["mentions"].transform(lambda s: s.rolling(7, min_periods=2).std())
    grouped["mentions_z_7"] = (grouped["mentions"] - grouped["mentions_ma_7"]) / grouped["mentions_std_7"]
    grouped["mentions_z_7"] = grouped["mentions_z_7"].replace([float("inf"), float("-inf")], pd.NA)
    grouped["mentions_z_7"] = grouped["mentions_z_7"].fillna(0.0)
    return grouped
