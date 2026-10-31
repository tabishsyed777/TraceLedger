"""Chronological training, evaluation, and scoring for synthetic payment data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

from traceledger.data import FEATURES, make_demo_data


@dataclass
class TraceLedgerModel:
    classifier: RandomForestClassifier
    evaluation: dict[str, float]
    feature_importance: pd.Series
    cutoff: pd.Timestamp


def fit_demo_model(seed: int = 19) -> tuple[TraceLedgerModel, pd.DataFrame]:
    data = make_demo_data(seed=29)
    split_at = int(len(data) * 0.80)
    cutoff = pd.Timestamp(data.loc[split_at, "timestamp"])
    train = data[data["timestamp"] < cutoff]
    test = data[data["timestamp"] >= cutoff]
    classifier = RandomForestClassifier(
        n_estimators=140, max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
    )
    classifier.fit(train[FEATURES], train["alert_truth"])
    probabilities = classifier.predict_proba(test[FEATURES])[:, 1]
    predicted = probabilities >= 0.5
    evaluation = {
        "test_transactions": float(len(test)),
        "test_alert_rate": float(test["alert_truth"].mean()),
        "average_precision": float(average_precision_score(test["alert_truth"], probabilities)),
        "precision_at_50": float(precision_score(test["alert_truth"], predicted, zero_division=0)),
        "recall_at_50": float(recall_score(test["alert_truth"], predicted, zero_division=0)),
        "train_transactions": float(len(train)),
        "train_incidents": float(train["alert_truth"].sum()),
    }
    model = TraceLedgerModel(
        classifier=classifier,
        evaluation=evaluation,
        feature_importance=pd.Series(classifier.feature_importances_, index=FEATURES).sort_values(ascending=False),
        cutoff=cutoff,
    )
    data["risk_score"] = classifier.predict_proba(data[FEATURES])[:, 1]
    return model, data


def score_transactions(model: TraceLedgerModel, data: pd.DataFrame) -> pd.DataFrame:
    scored = data.copy()
    scored["risk_score"] = model.classifier.predict_proba(scored[FEATURES])[:, 1]
    return scored


def reason_signals(row: pd.Series) -> list[str]:
    """Describe visible signals associated with a scored row; not causal explanations."""
    reasons = []
    if row["sender_txn_1h"] >= 3:
        reasons.append("High sender activity in the previous hour")
    if row["receiver_txn_1h"] >= 3:
        reasons.append("High receiver activity in the previous hour")
    if row["sender_unique_recipients_24h"] >= 4:
        reasons.append("Sender contacted several recipients in the previous day")
    if row["receiver_unique_senders_24h"] >= 4:
        reasons.append("Receiver collected from several senders in the previous day")
    if row["new_counterparty"]:
        reasons.append("No earlier sender-recipient pair in available history")
    if row["sender_amount_deviation"] >= 2.5:
        reasons.append("Amount differs from the sender's earlier transfer pattern")
    if not reasons:
        reasons.append("Combined model signals crossed the selected review threshold")
    return reasons
