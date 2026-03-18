from __future__ import annotations

import pandas as pd


def build_model_panel(reddit_daily: pd.DataFrame, market_df: pd.DataFrame, benchmark_df: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = reddit_daily.merge(market_df, on=["date", "ticker"], how="inner")

    if benchmark_df is not None and not benchmark_df.empty:
        bm = benchmark_df[["date", "return_1d", "next_return_1d"]].copy()
        bm = bm.rename(columns={
            "return_1d": "benchmark_return_1d",
            "next_return_1d": "benchmark_next_return_1d",
        })
        panel = panel.merge(bm, on="date", how="left")
        panel["next_abnormal_return_1d"] = panel["next_return_1d"] - panel["benchmark_next_return_1d"]
    else:
        panel["benchmark_return_1d"] = pd.NA
        panel["benchmark_next_return_1d"] = pd.NA
        panel["next_abnormal_return_1d"] = pd.NA

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["up_next_day"] = (panel["next_return_1d"] > 0).astype("Int64")
    keep_cols = [
        "date", "ticker", "mentions", "unique_authors", "avg_score", "total_score",
        "avg_sentiment", "frac_positive", "frac_negative", "avg_hype_score",
        "weighted_sentiment", "mentions_ma_7", "mentions_std_7", "mentions_z_7",
        "Adj Close", "Volume", "return_1d", "next_return_1d", "next_abs_return",
        "hl_range_pct", "next_hl_range_pct", "volume_change_1d", "next_volume_change_1d",
        "benchmark_return_1d", "benchmark_next_return_1d", "next_abnormal_return_1d",
        "up_next_day",
    ]
    return panel[[c for c in keep_cols if c in panel.columns]].copy()
