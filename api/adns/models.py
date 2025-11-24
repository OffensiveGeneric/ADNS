from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Flow(db.Model):
    __tablename__ = "flows"

    id = db.Column(db.Integer, primary_key=True)
    ts = db.Column(db.DateTime, nullable=False, index=True, default=datetime.utcnow)
    src_ip = db.Column(db.String(64), nullable=False)
    dst_ip = db.Column(db.String(64), nullable=False)
    proto = db.Column(db.String(16), nullable=False, default="tcp")
    bytes = db.Column(db.BigInteger, nullable=False, default=0)
    score = db.Column(db.Float, nullable=False, default=0.0)
    extra = db.Column(db.JSON, nullable=True)

    def to_dict(self):
        return {
            "ts": self.ts.isoformat() if self.ts else None,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "proto": self.proto,
            "bytes": self.bytes,
            "score": self.score,
            "extra": self.extra or {},
        }
