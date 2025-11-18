import os
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://adns:adns_password@127.0.0.1/adns"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

MAX_FLOWS = 200  # keep last N flows when responding to dashboard clients
FLOW_RETENTION_MINUTES = int(os.environ.get("ADNS_FLOW_RETENTION_MINUTES", "30"))
FLOW_RETENTION_MAX_ROWS = int(os.environ.get("ADNS_FLOW_RETENTION_MAX_ROWS", "5000"))

PROTOCOL_MAP = {
    "1": "ICMP",
    "6": "TCP",
    "17": "UDP",
    "41": "ENCAP",
    "47": "GRE",
    "50": "ESP",
    "51": "AH",
    "58": "ICMPv6",
    "132": "SCTP",
}


class Flow(db.Model):
    __tablename__ = "flows"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, index=True)
    src_ip = db.Column(db.String(64), nullable=False, index=True)
    dst_ip = db.Column(db.String(64), nullable=False, index=True)
    proto = db.Column(db.String(16), nullable=False)
    bytes = db.Column(db.Integer, nullable=False, default=0)

    predictions = db.relationship("Prediction", backref="flow", lazy="dynamic", cascade="all, delete-orphan")


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    flow_id = db.Column(db.Integer, db.ForeignKey("flows.id"), nullable=False, index=True)
    score = db.Column(db.Float, nullable=True)
    label = db.Column(db.String(32), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    with app.app_context():
        db.create_all()


def parse_timestamp(value) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            # allow trailing Z
            cleaned = value.replace("Z", "+00:00") if value.endswith("Z") else value
            return datetime.fromisoformat(cleaned)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def latest_prediction_score(flow: Flow) -> float:
    pred = flow.predictions.order_by(Prediction.created_at.desc()).first()
    if pred and pred.score is not None:
        return float(pred.score)
    return 0.0


def normalize_protocol(value) -> str:
    if value is None:
        return "OTHER"
    text = str(value).strip()
    if not text:
        return "OTHER"
    if text.isdigit():
        return PROTOCOL_MAP.get(text, f"PROTO_{text}")
    return text.upper()


def flow_to_dict(flow: Flow) -> dict:
    return {
        "id": flow.id,
        "ts": flow.timestamp.isoformat(),
        "src_ip": flow.src_ip,
        "dst_ip": flow.dst_ip,
        "proto": normalize_protocol(flow.proto),
        "bytes": flow.bytes,
        "score": latest_prediction_score(flow),
    }


def get_recent_flows(limit: int = MAX_FLOWS) -> list:
    flows = Flow.query.order_by(Flow.timestamp.desc()).limit(limit).all()
    # maintain chronological order (oldest first) for the dashboard
    return list(reversed(flows))


def enforce_flow_retention() -> int:
    purged = 0
    batch_size = 1000

    def delete_flow_batch(id_list: list[int]) -> int:
        if not id_list:
            return 0
        Prediction.query.filter(Prediction.flow_id.in_(id_list)).delete(synchronize_session=False)
        return Flow.query.filter(Flow.id.in_(id_list)).delete(synchronize_session=False)

    if FLOW_RETENTION_MINUTES > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=FLOW_RETENTION_MINUTES)
        while True:
            stale_ids = (
                Flow.query.with_entities(Flow.id)
                .filter(Flow.timestamp < cutoff)
                .limit(batch_size)
                .all()
            )
            id_list = [row.id for row in stale_ids]
            if not id_list:
                break
            purged += delete_flow_batch(id_list)

    if FLOW_RETENTION_MAX_ROWS > 0:
        total = Flow.query.count()
        if total > FLOW_RETENTION_MAX_ROWS:
            excess = total - FLOW_RETENTION_MAX_ROWS
            while excess > 0:
                chunk = min(excess, batch_size)
                oldest_ids = (
                    Flow.query.order_by(Flow.timestamp.asc())
                    .with_entities(Flow.id)
                    .limit(chunk)
                    .all()
                )
                id_list = [row.id for row in oldest_ids]
                if not id_list:
                    break
                purged += delete_flow_batch(id_list)
                excess -= len(id_list)

    if purged:
        db.session.commit()
    return purged


init_db()

# ---------------------------------------------------------------
# Basic Health Check
# ---------------------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------
# Ingest endpoint for tshark agent
# ---------------------------------------------------------------
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Accepts either:
      - a single flow object
      - a list of flow objects
    and persists them in the flows table.
    """
    payload = request.get_json(force=True, silent=False)

    if isinstance(payload, dict):
        batch = [payload]
    elif isinstance(payload, list):
        batch = payload
    else:
        return jsonify({"error": "invalid payload"}), 400

    created = 0
    for rec in batch:
        flow = Flow(
            timestamp=parse_timestamp(rec.get("ts")),
            src_ip=rec.get("src_ip", ""),
            dst_ip=rec.get("dst_ip", ""),
            proto=normalize_protocol(rec.get("proto", "")),
            bytes=int(rec.get("bytes") or 0),
        )
        db.session.add(flow)
        created += 1

    try:
        db.session.commit()
    except Exception as exc:  # pragma: no cover
        db.session.rollback()
        app.logger.exception("failed to insert flows: %s", exc)
        return jsonify({"error": "database insert failed"}), 500

    purged = enforce_flow_retention()
    if purged:
        app.logger.info("purged %d old flow(s)", purged)

    return jsonify({"status": "ok", "ingested": created, "purged": purged})


# ---------------------------------------------------------------
# Flows endpoint (dashboard)
#  - uses live buffer if present
#  - falls back to demo data if empty
# ---------------------------------------------------------------
@app.get("/flows")
def flows():
    recent = get_recent_flows()
    if recent:
        payload = [flow_to_dict(f) for f in recent]
        return jsonify(payload)

    demo_flows = [
        {
            "ts": "2025-11-17T11:10:00Z",
            "src_ip": "192.168.1.10",
            "dst_ip": "8.8.8.8",
            "proto": "TCP",
            "bytes": 1500,
            "score": 0.12,
        },
        {
            "ts": "2025-11-17T11:10:05Z",
            "src_ip": "10.0.0.5",
            "dst_ip": "172.217.3.110",
            "proto": "TCP",
            "bytes": 4200,
            "score": 0.98,
        },
        {
            "ts": "2025-11-17T11:10:09Z",
            "src_ip": "192.168.1.23",
            "dst_ip": "1.1.1.1",
            "proto": "UDP",
            "bytes": 800,
            "score": 0.45,
        },
    ]
    return jsonify(demo_flows)


# ---------------------------------------------------------------
# Anomaly stats (for now: simple derived stats from buffer or demo)
# ---------------------------------------------------------------
@app.get("/anomalies")
def anomalies():
    data = get_recent_flows()
    if not data:
        # same demo stats as before if nothing ingested yet
        demo_stats = {
            "window": "last 10 min",
            "count": 7,
            "max_score": 0.992,
            "pct_anomalous": 3.1,
        }
        return jsonify(demo_stats)

    scores = [latest_prediction_score(f) for f in data]
    total = len(scores)
    max_score = max(scores) if scores else 0.0
    anomaly_count = sum(1 for s in scores if s > 0.9)
    pct = (anomaly_count / total * 100.0) if total > 0 else 0.0

    stats = {
        "window": "recent buffer",
        "count": anomaly_count,
        "max_score": round(max_score, 3),
        "pct_anomalous": round(pct, 2),
    }
    return jsonify(stats)


# ---------------------------------------------------------------
# Main Entrypoint (for direct run; Gunicorn ignores this block)
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
