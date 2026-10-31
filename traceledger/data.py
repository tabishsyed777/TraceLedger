"""Synthetic payment network and causal time-window feature engineering."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

FEATURES = [
    "amount_log", "sender_txn_1h", "sender_txn_24h", "receiver_txn_1h",
    "receiver_txn_24h", "sender_outflow_24h_log", "receiver_inflow_24h_log",
    "sender_unique_recipients_24h", "receiver_unique_senders_24h",
    "new_counterparty", "sender_amount_deviation", "hour_sin", "hour_cos",
    "is_weekend", "channel_code",
]
REQUIRED = ["sender_id", "receiver_id", "timestamp", "amount"]
CHANNELS = {"card": 0, "ach": 1, "wire": 2, "cash": 3, "other": 4}


def make_demo_data(seed: int = 29, ordinary_count: int = 12000, incident_count: int = 24) -> pd.DataFrame:
    """Generate ordinary payments and labeled synthetic collection/dispersal incidents."""
    rng = np.random.default_rng(seed)
    account_count = 520
    accounts = [f"ACCT-{i:04d}" for i in range(account_count)]
    communities = {account: i // 65 for i, account in enumerate(accounts)}
    start = pd.Timestamp("2025-01-01", tz="UTC")
    horizon = pd.Timedelta(days=30)

    senders = rng.choice(accounts, size=ordinary_count)
    receivers = []
    for sender in senders:
        peer_pool = [account for account in accounts if communities[account] == communities[sender] and account != sender]
        if rng.random() < 0.78 and peer_pool:
            receivers.append(rng.choice(peer_pool))
        else:
            receivers.append(rng.choice([account for account in accounts if account != sender]))
    ordinary_times = start + pd.to_timedelta(rng.uniform(0, horizon.total_seconds(), ordinary_count), unit="s")
    amounts = np.clip(rng.lognormal(mean=3.5, sigma=0.9, size=ordinary_count), 2, 25000)
    channels = rng.choice(["card", "ach", "wire", "cash"], size=ordinary_count, p=[0.46, 0.34, 0.16, 0.04])

    rows = []
    for i in range(ordinary_count):
        rows.append({
            "transaction_id": f"TX-{i:07d}",
            "sender_id": str(senders[i]),
            "receiver_id": str(receivers[i]),
            "timestamp": ordinary_times[i],
            "amount": float(amounts[i]),
            "channel": str(channels[i]),
            "alert_truth": 0,
        })

    # Each simulated incident collects transfers at a hub, then rapidly disperses them.
    for incident in range(incident_count):
        center_time = start + pd.Timedelta(days=float(rng.uniform(1, 29)))
        participants = rng.choice(accounts, size=12, replace=False).tolist()
        hub, mule = participants[:2]
        sources, sinks = participants[2:7], participants[7:]
        incident_rows = []
        for source in sources:
            incident_rows.append((source, hub, center_time + pd.Timedelta(minutes=float(rng.uniform(-16, -3))), float(rng.uniform(150, 900))))
        incident_rows.append((hub, mule, center_time + pd.Timedelta(minutes=float(rng.uniform(0, 3))), float(rng.uniform(700, 4200))))
        for sink in sinks:
            incident_rows.append((mule, sink, center_time + pd.Timedelta(minutes=float(rng.uniform(4, 18))), float(rng.uniform(80, 700))))
        for offset, (sender, receiver, stamp, amount) in enumerate(incident_rows):
            rows.append({
                "transaction_id": f"SIM-{incident:03d}-{offset:02d}",
                "sender_id": sender,
                "receiver_id": receiver,
                "timestamp": stamp,
                "amount": amount,
                "channel": "wire" if rng.random() < 0.55 else "ach",
                "alert_truth": 1,
            })

    frame = pd.DataFrame(rows).sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    return engineer_features(frame)


def validate_transactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        return frame, [f"Missing required columns: {', '.join(missing)}"]
    data = frame.copy()
    if "channel" not in data.columns:
        data["channel"] = "other"
    if "transaction_id" not in data.columns:
        data["transaction_id"] = [f"UPLOAD-{i:07d}" for i in range(len(data))]
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data["sender_id"] = data["sender_id"].astype("string").str.strip()
    data["receiver_id"] = data["receiver_id"].astype("string").str.strip()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=REQUIRED)
    data = data[(data["amount"] > 0) & (data["sender_id"] != "") & (data["receiver_id"] != "")]
    if "alert_truth" in data:
        data = data.drop(columns="alert_truth")
    data = engineer_features(data)
    issues = []
    if data.empty:
        issues.append("No valid positive-amount transactions remain after cleaning.")
    elif len(data) < 30:
        issues.append("Upload at least 30 valid transactions for useful context features.")
    return data.reset_index(drop=True), issues


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create account history features using only records earlier than each transfer."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
    data = data.dropna(subset=REQUIRED)
    data = data[(data["amount"] > 0) & (data["sender_id"] != data["receiver_id"])]
    data = data.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)

    out_history = defaultdict(deque)
    in_history = defaultdict(deque)
    sender_stats = defaultdict(lambda: [0, 0.0, 0.0])
    seen_pairs = set()
    features = []

    for row in data.itertuples(index=False):
        sender, receiver = str(row.sender_id), str(row.receiver_id)
        stamp = pd.Timestamp(row.timestamp)
        now = stamp.timestamp()
        amount = float(row.amount)
        cutoff_day = now - 86400
        cutoff_hour = now - 3600
        outgoing = out_history[sender]
        incoming = in_history[receiver]
        while outgoing and outgoing[0][0] < cutoff_day:
            outgoing.popleft()
        while incoming and incoming[0][0] < cutoff_day:
            incoming.popleft()

        sender_hour = sum(event[0] >= cutoff_hour for event in outgoing)
        receiver_hour = sum(event[0] >= cutoff_hour for event in incoming)
        stats = sender_stats[sender]
        deviation = 0.0
        if stats[0] >= 3:
            std = (stats[2] / max(stats[0] - 1, 1)) ** 0.5
            deviation = abs(amount - stats[1]) / max(std, 1.0)
        hour = stamp.hour + stamp.minute / 60
        channel = str(getattr(row, "channel", "other")).lower()
        features.append({
            "amount_log": float(np.log1p(amount)),
            "sender_txn_1h": sender_hour,
            "sender_txn_24h": len(outgoing),
            "receiver_txn_1h": receiver_hour,
            "receiver_txn_24h": len(incoming),
            "sender_outflow_24h_log": float(np.log1p(sum(event[1] for event in outgoing))),
            "receiver_inflow_24h_log": float(np.log1p(sum(event[1] for event in incoming))),
            "sender_unique_recipients_24h": len({event[2] for event in outgoing}),
            "receiver_unique_senders_24h": len({event[2] for event in incoming}),
            "new_counterparty": int((sender, receiver) not in seen_pairs),
            "sender_amount_deviation": float(min(deviation, 20)),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "is_weekend": int(stamp.dayofweek >= 5),
            "channel_code": CHANNELS.get(channel, CHANNELS["other"]),
        })

        outgoing.append((now, amount, receiver))
        incoming.append((now, amount, sender))
        seen_pairs.add((sender, receiver))
        count, mean, m2 = stats
        count += 1
        delta = np.log1p(amount) - mean
        mean += delta / count
        m2 += delta * (np.log1p(amount) - mean)
        sender_stats[sender] = [count, mean, m2]

    feature_frame = pd.DataFrame(features, index=data.index)
    return pd.concat([data, feature_frame], axis=1)
