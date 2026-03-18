from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.aggregate_features import aggregate_reddit_features, ensure_datetime_date
from src.build_panel import build_model_panel
from src.collect_reddit import build_reddit_client, collect_comments, collect_submissions
from src.compute_sentiment import SentimentScorer, add_simple_hype_features
from src.detect_tickers import TickerDetector, build_ticker_universe
from src.download_prices import download_market_data
from src.model_baseline import run_logit, run_ols, save_logit_summary, save_regression_table, save_scatter_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Reddit-stock starter pipeline.")
    parser.add_argument("--subreddits", nargs="+", default=["wallstreetbets", "stocks"])
    parser.add_argument("--submission-limit", type=int, default=30)
    parser.add_argument("--comment-limit", type=int, default=50)
    parser.add_argument("--listing", choices=["hot", "new", "top"], default="hot")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--outputs-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    outputs_dir = Path(args.outputs_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "figures").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)

    reddit = build_reddit_client()
    submissions = collect_submissions(
        reddit=reddit,
        subreddits=args.subreddits,
        limit_per_subreddit=args.submission_limit,
        listing=args.listing,
    )
    comments = collect_comments(
        reddit=reddit,
        submission_ids=submissions["post_id"].tolist(),
        comment_limit_per_submission=args.comment_limit,
    )

    submissions.to_parquet(raw_dir / "reddit_submissions.parquet", index=False)
    comments.to_parquet(raw_dir / "reddit_comments.parquet", index=False)

    text_df = pd.concat([
        submissions[["source_type", "subreddit", "post_id", "created_utc", "author", "text", "score", "permalink"]],
        comments[["source_type", "subreddit", "post_id", "created_utc", "author", "text", "score", "permalink"]],
    ], ignore_index=True)

    ticker_universe = build_ticker_universe(args.tickers)
    detector = TickerDetector(ticker_universe=ticker_universe)
    mentions = detector.extract_from_frame(text_df, text_col="text")
    mentions = ensure_datetime_date(mentions, ts_col="created_utc")

    scorer = SentimentScorer()
    mentions = scorer.score_frame(mentions, text_col="text")
    mentions = add_simple_hype_features(mentions, text_col="text")
    mentions.to_parquet(processed_dir / "reddit_mentions_raw.parquet", index=False)

    reddit_daily = aggregate_reddit_features(mentions)
    reddit_daily.to_parquet(processed_dir / "reddit_mentions_daily.parquet", index=False)

    all_market = download_market_data(ticker_universe + ["SPY"], args.start_date, args.end_date)
    all_market.to_parquet(processed_dir / "market_data.parquet", index=False)

    benchmark_df = all_market[all_market["ticker"] == "SPY"].copy()
    market_df = all_market[all_market["ticker"] != "SPY"].copy()

    panel = build_model_panel(reddit_daily, market_df, benchmark_df)
    panel.to_parquet(processed_dir / "model_panel.parquet", index=False)

    ols_model = run_ols(panel, target="next_abs_return")
    save_regression_table(ols_model, outputs_dir / "tables" / "baseline_regression.csv")

    try:
        _, acc = run_logit(panel, target="up_next_day")
        save_logit_summary(acc, outputs_dir / "tables" / "logit_summary.txt")
        logit_msg = f"Logistic holdout accuracy: {acc:.4f}"
    except ValueError as e:
        save_logit_summary(float("nan"), outputs_dir / "tables" / "logit_summary.txt")
        logit_msg = f"Logistic model skipped: {e}"

    save_scatter_plot(panel, outputs_dir / "figures" / "mentions_vs_abs_return.png")

    print("Pipeline complete.")
    print(f"Rows in final panel: {len(panel)}")
    print(f"OLS R-squared: {ols_model.rsquared:.4f}")
    print(logit_msg)


if __name__ == "__main__":
    main()
