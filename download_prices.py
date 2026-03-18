from __future__ import annotations

from typing import Iterable

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def download_market_data(tickers: Iterable[str], start_date: str, end_date: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys([t.upper().strip() for t in tickers if str(t).strip()]))
    if not tickers:
        raise ValueError("No tickers provided.")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in tickers:
            if ticker not in data.columns.get_level_values(0):
                continue
            sub = data[ticker].copy()
            sub = sub.reset_index()
            sub["ticker"] = ticker
            frames.append(sub)
    else:
        sub = data.copy().reset_index()
        sub["ticker"] = tickers[0]
        frames.append(sub)

    if not frames:
        raise RuntimeError("No market data was downloaded.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={"Date": "date"})
    panel["date"] = pd.to_datetime(panel["date"]).dt.floor("D")

    for col in REQUIRED_PRICE_COLS:
        if col not in panel.columns:
            panel[col] = pd.NA

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["return_1d"] = panel.groupby("ticker")["Adj Close"].pct_change()
    panel["next_return_1d"] = panel.groupby("ticker")["return_1d"].shift(-1)
    panel["next_abs_return"] = panel["next_return_1d"].abs()
    panel["dollar_volume"] = panel["Adj Close"] * panel["Volume"]
    panel["hl_range_pct"] = (panel["High"] - panel["Low"]) / panel["Close"]
    panel["next_hl_range_pct"] = panel.groupby("ticker")["hl_range_pct"].shift(-1)
    panel["volume_change_1d"] = panel.groupby("ticker")["Volume"].pct_change()
    panel["next_volume_change_1d"] = panel.groupby("ticker")["volume_change_1d"].shift(-1)
    return panel
