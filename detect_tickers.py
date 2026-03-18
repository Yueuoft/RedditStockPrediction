from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

# Common finance/forum words that often create false ticker matches.
DEFAULT_BLOCKLIST = {
    "A", "I", "ALL", "ARE", "AI", "CEO", "CPI", "DD", "EPS", "ETF", "EV",
    "FDA", "FOMC", "GDP", "GM", "GOAT", "HODL", "IPO", "IRA", "IT", "LOL",
    "MD", "MOON", "NYSE", "OTC", "PE", "PM", "PR", "PT", "QQQ", "RH",
    "SEC", "SOFI", "SPAC", "TA", "TLDR", "USA", "WSB", "YOLO",
}

TICKER_PATTERN = re.compile(r"(?<![A-Z$])\$?[A-Z]{1,5}(?![A-Z])")


@dataclass(frozen=True)
class TickerDetectorConfig:
    min_mentions: int = 1
    require_valid_universe: bool = True
    allow_dollar_prefix_only_for_1_letter: bool = True


class TickerDetector:
    def __init__(
        self,
        ticker_universe: Iterable[str],
        blocklist: Iterable[str] | None = None,
        config: TickerDetectorConfig | None = None,
    ) -> None:
        self.ticker_universe = {t.upper() for t in ticker_universe}
        self.blocklist = set(DEFAULT_BLOCKLIST)
        if blocklist is not None:
            self.blocklist.update(x.upper() for x in blocklist)
        self.config = config or TickerDetectorConfig()

    def extract_from_text(self, text: str) -> list[str]:
        if not isinstance(text, str) or not text.strip():
            return []

        matches = TICKER_PATTERN.findall(text.upper())
        cleaned: list[str] = []

        for raw in matches:
            token = raw.replace("$", "")
            if not token:
                continue
            if token in self.blocklist:
                continue
            if len(token) == 1 and self.config.allow_dollar_prefix_only_for_1_letter and not raw.startswith("$"):
                continue
            if self.config.require_valid_universe and token not in self.ticker_universe:
                continue
            cleaned.append(token)

        # preserve order but deduplicate within one text row
        return list(dict.fromkeys(cleaned))

    def extract_from_frame(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        out = df.copy()
        out["tickers"] = out[text_col].apply(self.extract_from_text)
        out["mention_count_in_text"] = out["tickers"].str.len()
        out = out.explode("tickers", ignore_index=True)
        out = out.rename(columns={"tickers": "ticker"})
        out = out[out["ticker"].notna()].copy()
        return out


def build_ticker_universe(tickers: Iterable[str]) -> list[str]:
    return sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
