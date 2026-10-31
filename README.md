# TraceLedger

**Created by Tabish Syed**

TraceLedger is a synthetic transaction-monitoring workbench for exploring how graph and time-window signals can help prioritize payment records for analyst review.

It generates labeled synthetic payment activity, builds sender/receiver velocity and network features, trains a classifier with a chronological holdout, and presents ranked alerts with review notes. It does not connect to a bank, use customer data, or determine that a real person or transaction is criminal.

## Run locally

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Explore the demo

The generated dataset contains ordinary payments and simulated rapid collection-and-dispersal patterns. The evaluation uses a chronological split so later transactions are held out from training. The app shows precision, recall, average precision, model scores, and global feature importance.

Use the review queue to inspect scored transactions, record a disposition, and add an analyst note. Review actions remain in the current browser session and are not written to a database.

## Score a CSV

Upload a CSV containing:

| Column | Meaning |
| --- | --- |
| `sender_id` | Pseudonymous sending account |
| `receiver_id` | Pseudonymous receiving account |
| `timestamp` | ISO timestamp for the transfer |
| `amount` | Positive transfer amount |
| `channel` | Optional channel label such as wire, ACH, or card |

The app derives time-window counts, recent incoming/outgoing volume, counterpart novelty, and sender-relative amount deviation. Uploaded records are scored by a model trained on synthetic data; the resulting scores are not calibrated to real banking activity.

## Model and limits

A random forest classifier learns from generated examples labeled as ordinary or simulated rapid pass-through activity. Feature construction uses transaction history available up to each record. The held-out period provides a more realistic check than a random row split, but it remains a synthetic benchmark.

An alert is a triage signal for review, not a finding of fraud, money laundering, sanctions exposure, or wrongdoing. Real transaction monitoring requires institution-specific data, policy controls, privacy and security review, fairness testing, human investigation, and regulatory expertise.

## License

MIT. See [LICENSE](LICENSE).
