from __future__ import annotations

import pandas as pd
import streamlit as st

from traceledger.data import FEATURES, make_demo_data, validate_transactions
from traceledger.model import fit_demo_model, reason_signals, score_transactions

st.set_page_config(page_title="TraceLedger", page_icon="◈", layout="wide")


@st.cache_resource
def load_model():
    return fit_demo_model()


model, demo_scored = load_model()
st.title("◈ TraceLedger")
st.caption("Synthetic payment network monitoring · explainable alert triage · analyst review")
st.info(
    "Synthetic portfolio demo only. Risk scores rank records for review; they are not "
    "findings of fraud, money laundering, sanctions exposure, or wrongdoing."
)

with st.sidebar:
    st.header("Workspace")
    uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
    st.caption("Required fields: sender_id, receiver_id, timestamp, amount.")
    threshold = st.slider("Review threshold", 0.05, 0.95, 0.55, 0.05)
    st.caption("Threshold is a demonstration control, not an operational policy.")

if uploaded is None:
    data = demo_scored.copy()
    st.caption("Synthetic demo network · scenario labels are used only for the held-out evaluation.")
else:
    try:
        raw = pd.read_csv(uploaded)
        clean, issues = validate_transactions(raw)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()
    if issues:
        for issue in issues:
            st.error(issue)
        st.stop()
    data = score_transactions(model, clean)
    st.caption(f"Scored {len(data):,} valid transactions across {data['sender_id'].nunique()} senders.")

if "review_state" not in st.session_state:
    st.session_state.review_state = {}
if "review_notes" not in st.session_state:
    st.session_state.review_notes = {}

queue = data.sort_values("risk_score", ascending=False).copy()
queue["review_state"] = queue["transaction_id"].map(st.session_state.review_state).fillna("New")
queue["review_flag"] = queue["risk_score"] >= threshold
alerts = queue[queue["review_flag"]]
reviewed_count = sum(value != "New" for value in st.session_state.review_state.values())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions", f"{len(data):,}")
c2.metric("Priority queue", f"{len(alerts):,}", f"score ≥ {threshold:.2f}")
c3.metric("Reviewed this session", f"{reviewed_count:,}")
c4.metric("Accounts", f"{len(set(data['sender_id']) | set(data['receiver_id'])):,}")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Recent payment activity")
    by_day = data.set_index("timestamp").sort_index().resample("D").agg(
        Transactions=("transaction_id", "count"),
        "High-score records"=("risk_score", lambda x: int((x >= threshold).sum())),
    )
    st.line_chart(by_day, height=280)
with right:
    st.subheader("Model evaluation")
    ev = model.evaluation
    a, b = st.columns(2)
    a.metric("Average precision", f"{ev['average_precision']:.3f}")
    b.metric("Recall at 0.50", f"{ev['recall_at_50']:.1%}")
    st.caption(
        f"Chronological holdout: {int(ev['test_transactions']):,} transactions · "
        f"simulated positive rate {ev['test_alert_rate']:.2%}"
    )
    st.caption("Synthetic benchmark only; scores are not calibrated for real payment data.")

st.subheader("Alert review queue")
if alerts.empty:
    st.success("No rows cross the selected threshold.")
else:
    display_columns = [
        "transaction_id", "timestamp", "sender_id", "receiver_id", "amount",
        "channel", "risk_score", "review_state",
    ]
    st.dataframe(
        alerts[display_columns].head(100).style.format({"amount": "{:,.2f}", "risk_score": "{:.1%}"}),
        use_container_width=True,
    )

transaction_ids = alerts["transaction_id"].astype(str).head(100).tolist()
if transaction_ids:
    selected_id = st.selectbox("Inspect a transaction", transaction_ids)
    selected = queue[queue["transaction_id"].astype(str) == selected_id].iloc[0]
    detail, action = st.columns([1.2, 1])
    with detail:
        st.markdown(f"**{selected['sender_id']} → {selected['receiver_id']}**")
        st.write(
            f"{selected['amount']:,.2f} · {selected['channel']} · "
            f"{selected['timestamp']} · score {selected['risk_score']:.1%}"
        )
        st.markdown("**Signals to inspect**")
        for reason in reason_signals(selected):
            st.write(f"• {reason}")
        st.caption("These are visible signals, not a causal explanation or a conclusion.")
    with action:
        with st.form(f"review-{selected_id}"):
            state = st.selectbox(
                "Disposition",
                ["New", "Investigate", "Escalate", "Close - no concern"],
                index=["New", "Investigate", "Escalate", "Close - no concern"].index(
                    st.session_state.review_state.get(selected_id, "New")
                ),
            )
            note = st.text_area("Analyst note", value=st.session_state.review_notes.get(selected_id, ""))
            submitted = st.form_submit_button("Save review")
            if submitted:
                st.session_state.review_state[selected_id] = state
                st.session_state.review_notes[selected_id] = note
                st.rerun()

with st.expander("Global model signals"):
    st.bar_chart(model.feature_importance.head(10), horizontal=True)
    st.write("Features used:", ", ".join(FEATURES))

with st.expander("Evaluation and data notes"):
    st.write(
        "Features are computed in timestamp order from earlier sender and receiver activity. "
        "The held-out chronological period is not used to fit the classifier."
    )
    st.write(
        "Review dispositions and notes are kept only in Streamlit's current session state. "
        "They are not persisted to a database or sent elsewhere."
    )
