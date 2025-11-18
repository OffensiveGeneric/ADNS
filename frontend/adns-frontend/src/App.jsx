import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

export default function App() {
  const [flows, setFlows] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [srcFilter, setSrcFilter] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        setError("");
        const [flowsRes, statsRes] = await Promise.all([
          axios.get("/api/flows"),
          axios.get("/api/anomalies"),
        ]);
        setFlows(flowsRes.data || []);
        setStats(statsRes.data || null);
      } catch (err) {
        console.error(err);
        setError("Unable to load data from API");
      } finally {
        setLoading(false);
      }
    };

    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  const sortedFlows = useMemo(() => {
    return [...flows].sort((a, b) => new Date(b.ts) - new Date(a.ts));
  }, [flows]);

  const visibleFlows = useMemo(() => {
    if (!srcFilter.trim()) {
      return sortedFlows;
    }
    const needle = srcFilter.trim().toLowerCase();
    return sortedFlows.filter((flow) =>
      (flow.src_ip || "").toLowerCase().includes(needle),
    );
  }, [sortedFlows, srcFilter]);

  const chartData = sortedFlows.map((f, i) => ({
    index: i,
    score: f.score,
  }));

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Live security telemetry</p>
          <h1>ADNS Dashboard</h1>
          <p className="app-subtitle">
            Anomaly Detection Network System — live traffic overview
          </p>
        </div>
      </header>

      {error && <div className="app-alert">{error}</div>}

      <section className="metrics-grid">
        <Card
          title="Anomalies (10 min)"
          value={stats?.count ?? (loading ? "…" : "0")}
        />
        <Card
          title="Max anomaly score"
          value={
            stats?.max_score != null
              ? stats.max_score.toFixed(3)
              : loading
              ? "…"
              : "0.000"
          }
        />
        <Card
          title="% traffic anomalous"
          value={
            stats?.pct_anomalous != null
              ? `${stats.pct_anomalous}%`
              : loading
              ? "…"
              : "0%"
          }
        />
      </section>

      <section className="panel chart-panel">
        <div className="panel-heading">
          <h3>Anomaly score over recent flows</h3>
        </div>
        {chartData.length === 0 ? (
          <p className="empty-state">No flow data yet.</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="index" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="score"
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      <section className="panel table-panel">
        <div className="panel-heading">
          <h3>Recent flows</h3>
        </div>
        {visibleFlows.length === 0 ? (
          <p className="empty-state">No flows yet.</p>
        ) : (
          <>
            <div className="filter-row">
              <label htmlFor="srcFilter">Filter by source IP</label>
              <input
                id="srcFilter"
                type="text"
                value={srcFilter}
                onChange={(e) => setSrcFilter(e.target.value)}
                placeholder="e.g. 192.168"
              />
            </div>
            <div className="table-wrapper">
              <table className="flow-table">
                <thead>
                  <tr>
                    <Th>Time</Th>
                    <Th>Source IP</Th>
                    <Th>Destination IP</Th>
                    <Th>Proto</Th>
                    <Th>Bytes</Th>
                    <Th>Score</Th>
                  </tr>
                </thead>
                <tbody>
                  {visibleFlows.map((f, idx) => (
                    <tr key={idx}>
                      <Td>{f.ts}</Td>
                      <Td>{f.src_ip}</Td>
                      <Td>{f.dst_ip}</Td>
                      <Td>{f.proto}</Td>
                      <Td>{f.bytes}</Td>
                      <Td>
                        <ScoreTag score={f.score} />
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function Card({ title, value }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
    </div>
  );
}

function Th({ children }) {
  return <th>{children}</th>;
}

function Td({ children }) {
  return <td>{children}</td>;
}

function ScoreTag({ score }) {
  const s = Number(score) || 0;
  let bg = "#e8f5e9";
  let color = "#1b5e20";
  if (s > 0.9) {
    bg = "#ffebee";
    color = "#b71c1c";
  } else if (s > 0.6) {
    bg = "#fff3e0";
    color = "#e65100";
  }
  return (
    <span
      className="score-tag"
      style={{
        background: bg,
        color,
      }}
    >
      {s.toFixed(3)}
    </span>
  );
}
