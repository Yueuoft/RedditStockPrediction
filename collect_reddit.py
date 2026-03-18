from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import praw


def build_reddit_client() -> praw.Reddit:
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT")

    missing = [name for name, value in {
        "REDDIT_CLIENT_ID": client_id,
        "REDDIT_CLIENT_SECRET": client_secret,
        "REDDIT_USER_AGENT": user_agent,
    }.items() if not value]
    if missing:
        raise EnvironmentError(
            "Missing Reddit API environment variables: " + ", ".join(missing)
        )

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )


def collect_submissions(
    reddit: praw.Reddit,
    subreddits: Iterable[str],
    limit_per_subreddit: int,
    listing: str = "hot",
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sr in subreddits:
        subreddit = reddit.subreddit(sr)
        if listing == "new":
            iterator = subreddit.new(limit=limit_per_subreddit)
        elif listing == "top":
            iterator = subreddit.top(limit=limit_per_subreddit, time_filter="year")
        else:
            iterator = subreddit.hot(limit=limit_per_subreddit)

        for submission in iterator:
            rows.append({
                "source_type": "submission",
                "subreddit": sr,
                "post_id": submission.id,
                "parent_post_id": submission.id,
                "created_utc": pd.to_datetime(submission.created_utc, unit="s", utc=True),
                "author": getattr(submission.author, "name", "[deleted]"),
                "title": submission.title,
                "selftext": submission.selftext,
                "text": f"{submission.title or ''} {submission.selftext or ''}".strip(),
                "score": submission.score,
                "num_comments": submission.num_comments,
                "url": submission.url,
                "permalink": f"https://www.reddit.com{submission.permalink}",
            })
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return pd.DataFrame(rows)


def collect_comments(
    reddit: praw.Reddit,
    submission_ids: Iterable[str],
    comment_limit_per_submission: int,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sid in submission_ids:
        submission = reddit.submission(id=sid)
        submission.comments.replace_more(limit=0)
        comments = submission.comments.list()[:comment_limit_per_submission]
        for comment in comments:
            rows.append({
                "source_type": "comment",
                "subreddit": submission.subreddit.display_name,
                "comment_id": comment.id,
                "post_id": sid,
                "parent_post_id": sid,
                "created_utc": pd.to_datetime(comment.created_utc, unit="s", utc=True),
                "author": getattr(comment.author, "name", "[deleted]"),
                "text": comment.body,
                "score": comment.score,
                "permalink": f"https://www.reddit.com{comment.permalink}",
            })
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Reddit submissions and comments.")
    parser.add_argument("--subreddits", nargs="+", required=True)
    parser.add_argument("--submission-limit", type=int, default=30)
    parser.add_argument("--comment-limit", type=int, default=50)
    parser.add_argument("--listing", choices=["hot", "new", "top"], default="hot")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    reddit = build_reddit_client()
    submissions = collect_submissions(
        reddit=reddit,
        subreddits=args.subreddits,
        limit_per_subreddit=args.submission_limit,
        listing=args.listing,
        sleep_seconds=args.sleep_seconds,
    )
    submissions.to_parquet(raw_dir / "reddit_submissions.parquet", index=False)

    comments = collect_comments(
        reddit=reddit,
        submission_ids=submissions["post_id"].tolist(),
        comment_limit_per_submission=args.comment_limit,
        sleep_seconds=args.sleep_seconds,
    )
    comments.to_parquet(raw_dir / "reddit_comments.parquet", index=False)

    print(f"Saved {len(submissions)} submissions and {len(comments)} comments to {raw_dir}")


if __name__ == "__main__":
    main()
