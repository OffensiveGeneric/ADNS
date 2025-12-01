import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  AreaChart,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import "./App.css";

const apiBase = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
// If no base is provided, rely on Vite dev proxy (/api -> http://127.0.0.1:5000)
const api = axios.create({ baseURL: apiBase });

const SIM_ATTACKS = [
  {
    type: "botnet_flood",
    label: "Botnet Flood",
    description: "IoT swarm saturating a target",
  },
  {
    type: "data_exfiltration",
    label: "Data Exfiltration",
    description: "Large outbound transfer",
  },
  {
    type: "port_scan",
    label: "Port Scan",
    description: "Rapid lateral probing",
  },
];

const severityFromLabel = (label, score) => {
  const normalized = (label || "").toLowerCase();
  if (normalized === "anomaly" || normalized === "high") {
    return "anomaly";
  }
  if (normalized === "watch" || normalized === "medium") {
    return "watch";
  }
  if (normalized === "normal" || normalized === "low") {
    return "normal";
  }
  const s = Number(score) || 0;
  if (s >= 0.9) return "anomaly";
  if (s >= 0.6) return "watch";
  return "normal";
};

export default function App() {
  const [flows, setFlows] = useState([]);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [srcFilter, setSrcFilter] = useState("");
  const [simBusy, setSimBusy] = useState("");
  const [simStatus, setSimStatus] = useState(null);

  const fetchLatest = useCallback(async () => {
    try {
      setError("");
      const [flowsRes, statsRes] = await Promise.all([
        api.get("/api/flows"),
        api.get("/api/anomalies"),
      ]);
      const fetchedFlows = flowsRes.data || [];
      setFlows(fetchedFlows);
      setStats(statsRes.data || null);
    } catch (err) {
      console.error(err);
      setError("Unable to load data from API");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLatest();
    const id = setInterval(fetchLatest, 2000);
    return () => clearInterval(id);
  }, [fetchLatest]);

  const triggerSimulation = async (attack) => {
    setSimBusy(attack.type);
    setSimStatus({
      tone: "info",
      message: `Triggering ${attack.label}…`,
    });
    try {
      const resp = await api.post("/api/simulate", { type: attack.type });
      await fetchLatest();
      setTimeout(fetchLatest, 1000);
      setSimStatus({
        tone: "success",
        message: `Generated ${resp.data.generated} flows (${attack.label}).`,
      });
    } catch (err) {
      console.error(err);
      const apiError = err?.response?.data?.error ?? "server error";
      setSimStatus({
        tone: "error",
        message: `Failed to trigger ${attack.label}: ${apiError}`,
      });
    } finally {
      setSimBusy("");
    }
  };

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

  const timelineData = useMemo(() => {
    const ordered = [...sortedFlows].reverse();
    const recent = ordered.slice(-30);
    return recent.map((flow) => ({
      tsLabel: new Date(flow.ts).toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      score: Number(flow.score) || 0,
      severity: severityFromLabel(flow.label, flow.score),
    }));
  }, [sortedFlows]);

  const severityCounts = useMemo(() => {
    return sortedFlows.reduce(
      (acc, flow) => {
        const severity = severityFromLabel(flow.label, flow.score);
        acc[severity] += 1;
        return acc;
      },
      { anomaly: 0, watch: 0, normal: 0 },
    );
  }, [sortedFlows]);

  const donutData = [
    { name: "Anomaly", value: severityCounts.anomaly, severity: "anomaly" },
    { name: "Watch", value: severityCounts.watch, severity: "watch" },
    { name: "Normal", value: severityCounts.normal, severity: "normal" },
  ];

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

      <section className="panel simulation-panel">
        <div className="panel-heading">
          <h3>Attack simulation controls</h3>
          <p>Use these demo buttons to stream synthetic malicious traffic.</p>
        </div>
        <div className="simulate-grid">
          {SIM_ATTACKS.map((attack) => (
            <button
              key={attack.type}
              type="button"
              className={`simulate-btn${
                simBusy === attack.type ? " is-active" : ""
              }`}
              onClick={() => triggerSimulation(attack)}
              disabled={Boolean(simBusy)}
            >
              <span>{attack.label}</span>
              <small>{attack.description}</small>
            </button>
          ))}
        </div>
        {simStatus?.message && (
          <p className={`simulate-status ${simStatus.tone}`}>
            {simStatus.message}
          </p>
        )}
      </section>

      <section className="panel timeline-panel">
        <div className="panel-heading">
          <h3>Threat timeline</h3>
          <p>Recent flow scores with severity shading.</p>
        </div>
        {timelineData.length === 0 ? (
          <p className="empty-state">Timeline will appear once flows arrive.</p>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={timelineData}>
              <defs>
                <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.7} />
                  <stop offset="40%" stopColor="#f97316" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0.2} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" vertical={false} />
              <XAxis
                dataKey="tsLabel"
                tick={{ fontSize: 10 }}
                interval={timelineData.length > 12 ? 2 : 0}
              />
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                formatter={(value, _, entry) => [
                  (Number(value) || 0).toFixed(3),
                  entry.payload.severity,
                ]}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke="#ea580c"
                fill="url(#scoreGradient)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </section>

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

      <section className="panel donut-panel">
        <div className="panel-heading">
          <h3>Severity mix</h3>
          <p>Breakdown of recent flows by model decision.</p>
        </div>
        <div className="donut-wrapper">
          <ResponsiveContainer width="55%" height={220}>
            <PieChart>
              <Pie
                data={donutData}
                innerRadius={60}
                outerRadius={90}
                dataKey="value"
                paddingAngle={2}
              >
                {donutData.map((entry) => (
                  <Cell key={entry.severity} fill={threatColor(entry.severity)} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name) => [`${value} flows`, name]}
              />
            </PieChart>
          </ResponsiveContainer>
          <ul className="donut-legend">
            {donutData.map((entry) => (
              <li key={entry.severity}>
                <span
                  className="dot"
                  style={{ background: threatColor(entry.severity) }}
                />
                {entry.name}: {entry.value}
              </li>
            ))}
          </ul>
        </div>
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
                    <Th>Severity</Th>
                  </tr>
                </thead>
                <tbody>
                  {visibleFlows.map((f, idx) => (
                    <tr key={idx}>
                      <Td>{new Date(f.ts).toLocaleString()}</Td>
                      <Td>{f.src_ip}</Td>
                      <Td>{f.dst_ip}</Td>
                      <Td>{f.proto}</Td>
                      <Td>{f.bytes}</Td>
                      <Td clamp={false}>
                        <ScoreTag score={f.score} />
                      </Td>
                      <Td clamp={false}>
                        <ThreatBadge label={f.label} score={f.score} />
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

function Td({ children, clamp = true }) {
  if (!clamp) {
    return <td>{children}</td>;
  }
  return (
    <td>
      <span className="cell-text">{children}</span>
    </td>
  );
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

function ThreatBadge({ label, score }) {
  const severity = severityFromLabel(label, score);
  const config = severityConfig();
  const { text, color, bg, icon } = config[severity] || config.normal;
  return (
    <span
      className="threat-badge"
      style={{
        color,
        background: bg,
      }}
    >
      <span className="icon">{icon}</span>
      {text}
    </span>
  );
}

function threatColor(severity) {
  const config = severityConfig();
  return config[severity]?.color ?? config.normal.color;
}

function severityConfig() {
  return {
    anomaly: {
      text: "Anomaly",
      color: "#b91c1c",
      bg: "#fee2e2",
      icon: "⚠️",
    },
    watch: {
      text: "Watch",
      color: "#b45309",
      bg: "#fff7ed",
      icon: "👁️",
    },
    normal: {
      text: "Normal",
      color: "#166534",
      bg: "#e0f2fe",
      icon: "✅",
    },
  };
}
