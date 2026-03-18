from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "mentions",
    "unique_authors",
    "avg_score",
    "avg_sentiment",
    "frac_positive",
    "frac_negative",
    "avg_hype_score",
    "weighted_sentiment",
    "mentions_z_7",
    "return_1d",
    "hl_range_pct",
    "volume_change_1d",
]


def prepare_model_data(panel: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = FEATURES + [target]
    df = panel[cols].copy()
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return df


def run_ols(panel: pd.DataFrame, target: str = "next_abs_return") -> sm.regression.linear_model.RegressionResultsWrapper:
    df = prepare_model_data(panel, target)
    X = sm.add_constant(df[FEATURES])
    y = df[target]
    model = sm.OLS(y, X).fit(cov_type="HC3")
    return model


def run_logit(panel: pd.DataFrame, target: str = "up_next_day") -> tuple[Pipeline, float]:
    df = prepare_model_data(panel, target)
    X = df[FEATURES]
    y = df[target].astype(int)

    if y.nunique() < 2:
        raise ValueError("Target for logistic regression has fewer than 2 classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    return pipe, acc


def save_regression_table(model: sm.regression.linear_model.RegressionResultsWrapper, output_csv: str) -> None:
    table = pd.DataFrame({
        "coef": model.params,
        "std_err": model.bse,
        "t": model.tvalues,
        "p_value": model.pvalues,
    })
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=True)


def save_logit_summary(acc: float, output_txt: str) -> None:
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Holdout accuracy: {acc:.4f}\n")


def save_scatter_plot(panel: pd.DataFrame, output_png: str) -> None:
    df = panel[["mentions", "next_abs_return"]].replace([float("inf"), float("-inf")], pd.NA).dropna()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.scatter(df["mentions"], df["next_abs_return"], alpha=0.6)
    plt.xlabel("Daily Reddit mentions")
    plt.ylabel("Next-day absolute return")
    plt.title("Reddit mentions vs next-day absolute return")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
