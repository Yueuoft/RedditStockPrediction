# Reddit Sentiment and Stock Activity Starter Project

This is a complete starter package for a mini project on whether Reddit attention and sentiment predict stock returns, volatility, or volume.

## What this package does

1. Collects Reddit submissions and comments with `praw`.
2. Extracts valid stock tickers using a ticker universe plus simple ambiguity filters.
3. Scores sentiment using VADER.
4. Aggregates Reddit data to a ticker-date panel.
5. Downloads market data with `yfinance`.
6. Merges Reddit features with stock data.
7. Runs a baseline regression and classification model.
8. Saves tables and a figure.

## Recommended research question

Do Reddit hype and sentiment predict next-day abnormal volatility or next-day returns for heavily discussed stocks?

## Folder structure

```text
reddit_stock_project/
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── tables/
├── src/
│   ├── collect_reddit.py
│   ├── detect_tickers.py
│   ├── compute_sentiment.py
│   ├── aggregate_features.py
│   ├── download_prices.py
│   ├── build_panel.py
│   └── model_baseline.py
├── requirements.txt
└── run_pipeline.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Reddit API setup

Create a Reddit app and then set these environment variables:

```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
export REDDIT_USER_AGENT="reddit-stock-project by /u/your_username"
```

On Windows PowerShell:

```powershell
$env:REDDIT_CLIENT_ID="your_client_id"
$env:REDDIT_CLIENT_SECRET="your_client_secret"
$env:REDDIT_USER_AGENT="reddit-stock-project by /u/your_username"
```

## Quick start

This example collects about 30 hot submissions from `wallstreetbets` and `stocks`, then fetches comments for those threads, builds the dataset, and runs the model.

```bash
python run_pipeline.py \
  --subreddits wallstreetbets stocks \
  --submission-limit 30 \
  --comment-limit 50 \
  --tickers AAPL MSFT NVDA TSLA AMD META AMZN GOOGL NFLX SPY \
  --start-date 2025-01-01 \
  --end-date 2025-12-31
```

## Outputs

After running, you should get:

- `data/raw/reddit_submissions.parquet`
- `data/raw/reddit_comments.parquet`
- `data/processed/reddit_mentions_daily.parquet`
- `data/processed/market_data.parquet`
- `data/processed/model_panel.parquet`
- `outputs/tables/baseline_regression.csv`
- `outputs/tables/logit_summary.txt`
- `outputs/figures/mentions_vs_abs_return.png`

## Notes

- This is a starter project, not a production pipeline.
- Reddit API access is rate-limited, so start small.
- For a stronger paper, switch the target from signed return to `next_abs_return` or `next_day_range_pct`.
- The ticker extraction logic is conservative on purpose to reduce false positives.

## Good next upgrades

1. Add a custom WallStreetBets hype dictionary.
2. Restrict Reddit timestamps to pre-market-close information.
3. Add event studies for mention spikes.
4. Replace VADER with FinBERT.
5. Move heavy text preprocessing into PySpark.
