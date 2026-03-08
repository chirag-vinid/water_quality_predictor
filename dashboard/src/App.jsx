import { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

const WS_URL      = "ws://localhost:8000/ws";
const HISTORY_URL = "http://localhost:8000/history?n=60";
const MAX_POINTS  = 60;

const wqiColor = (wqi) => {
  if (wqi === null || wqi === undefined) return "#4a5568";
  if (wqi >= 80) return "#00e5a0";
  if (wqi >= 60) return "#f5c542";
  if (wqi >= 50) return "#ff8c42";
  return "#ff3b5c";
};

const wqiLabel = (wqi) => {
  if (wqi === null || wqi === undefined) return "—";
  if (wqi >= 80) return "Excellent";
  if (wqi >= 60) return "Good";
  if (wqi >= 50) return "Fair";
  return "Poor";
};

const fmtTime = (ts) => {
  if (!ts) return "";
  return new Date(ts).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
};

function PulseRing({ color }) {
  return (
    <span style={{ position: "relative", display: "inline-block", width: 12, height: 12 }}>
      <span style={{
        position: "absolute", inset: 0, borderRadius: "50%",
        background: color, animation: "ping 1.5s ease-out infinite", opacity: 0.5,
      }} />
      <span style={{ position: "absolute", inset: "2px", borderRadius: "50%", background: color }} />
    </span>
  );
}

function StatCard({ label, value, unit, color, sub }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: `1px solid rgba(255,255,255,0.07)`,
      borderTop: `2px solid ${color}`,
      borderRadius: 8,
      padding: "14px 18px",
      flex: 1,
      minWidth: 140,
    }}>
      <div style={{ fontSize: 11, color: "#6b7a8d", fontFamily: "'Courier New', monospace", letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 700, color, marginTop: 4, fontFamily: "'Bebas Neue', cursive" }}>
        {value ?? "—"}<span style={{ fontSize: 13, color: "#6b7a8d", marginLeft: 3 }}>{unit}</span>
      </div>
      {sub && <div style={{ fontSize: 11, color: "#4a5568", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function WQIGauge({ value }) {
  const color = wqiColor(value);
  const label = wqiLabel(value);
  const pct   = value ? Math.min(Math.max((value - 37) / (95 - 37), 0), 1) : 0;
  const angle = -135 + pct * 270;

  return (
    <div style={{ textAlign: "center", padding: "10px 0" }}>
      <svg viewBox="0 0 200 120" width="180" height="110">
        <path d="M 20 110 A 80 80 0 1 1 180 110" fill="none" stroke="#1e2633" strokeWidth="14" strokeLinecap="round" />
        <path d="M 20 110 A 80 80 0 1 1 180 110" fill="none" stroke={color}
          strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${pct * 251.2} 251.2`} style={{ transition: "all 0.6s ease" }} />
        <g transform={`rotate(${angle}, 100, 110)`}>
          <line x1="100" y1="110" x2="100" y2="40" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="100" cy="110" r="5" fill={color} />
        </g>
        <text x="100" y="98" textAnchor="middle" fill={color}
          style={{ fontSize: 28, fontFamily: "'Bebas Neue', cursive", fontWeight: 700 }}>
          {value ? value.toFixed(1) : "—"}
        </text>
        <text x="100" y="114" textAnchor="middle" fill="#6b7a8d" style={{ fontSize: 10 }}>
          {label}
        </text>
      </svg>
    </div>
  );
}

function SensorBar({ label, value, min, max, unit, color }) {
  const pct = value !== null ? Math.min(Math.max((value - min) / (max - min), 0), 1) * 100 : 0;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span style={{ fontSize: 11, color: "#6b7a8d", fontFamily: "'Courier New', monospace" }}>{label}</span>
        <span style={{ fontSize: 12, color, fontWeight: 600 }}>
          {value?.toFixed(1) ?? "—"} <span style={{ color: "#4a5568", fontSize: 10 }}>{unit}</span>
        </span>
      </div>
      <div style={{ height: 5, background: "#1e2633", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%", width: `${pct}%`, background: color,
          borderRadius: 3, transition: "width 0.6s ease",
        }} />
      </div>
    </div>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0d1117", border: "1px solid #2d3748",
      borderRadius: 6, padding: "8px 12px", fontSize: 12, color: "#cbd5e0",
    }}>
      <div style={{ color: "#6b7a8d", marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: <strong>{p.value?.toFixed(1)}</strong>
        </div>
      ))}
    </div>
  );
};

export default function App() {
  const [readings,   setReadings]   = useState([]);
  const [latest,     setLatest]     = useState(null);
  const [connected,  setConnected]  = useState(false);
  const [lastUpdate, setLastUpdate] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    fetch(HISTORY_URL)
      .then(r => r.json())
      .then(data => {
        const pts = (data.readings || []).map(r => ({
          time:      fmtTime(r.timestamp),
          rfWqi:     r.rf?.wqi ?? null,
          lstmWqi15: r.lstm?.wqi_15min ?? null,
          lstmWqi30: r.lstm?.wqi_30min ?? null,
          lstmWqi60: r.lstm?.wqi_60min ?? null,
          gtWqi:     r.ground_truth_wqi ?? null,
          pH:        r.sensor?.pH ?? null,
          turbidity: r.sensor?.Turbidity_NTU ?? null,
          tds:       r.sensor?.TDS_ppm ?? null,
          temp:      r.sensor?.Temperature_C ?? null,
          flow:      r.sensor?.Flow_Rate_Lmin ?? null,
        }));
        setReadings(pts);
        if (pts.length) setLatest(data.readings[data.readings.length - 1]);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => { setConnected(false); setTimeout(connect, 3000); };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        const r = JSON.parse(e.data);
        setLatest(r);
        setLastUpdate(new Date().toLocaleTimeString());
        setReadings(prev => [...prev, {
          time:      fmtTime(r.timestamp),
          rfWqi:     r.rf?.wqi ?? null,
          lstmWqi15: r.lstm?.wqi_15min ?? null,
          lstmWqi30: r.lstm?.wqi_30min ?? null,
          lstmWqi60: r.lstm?.wqi_60min ?? null,
          gtWqi:     r.ground_truth_wqi ?? null,
          pH:        r.sensor?.pH ?? null,
          turbidity: r.sensor?.Turbidity_NTU ?? null,
          tds:       r.sensor?.TDS_ppm ?? null,
          temp:      r.sensor?.Temperature_C ?? null,
          flow:      r.sensor?.Flow_Rate_Lmin ?? null,
        }].slice(-MAX_POINTS));
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const sensor  = latest?.sensor ?? {};
  const rf      = latest?.rf ?? {};
  const lstm    = latest?.lstm ?? {};
  const rfColor = wqiColor(rf.wqi);

  return (
    <div style={{
      width: "100%",
      minHeight: "100vh",
      background: "#080c12",
      color: "#e2e8f0",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      padding: "20px 32px",
      boxSizing: "border-box",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet" />
      <style>{`
        @keyframes ping {
          0%   { transform: scale(1);   opacity: .5; }
          100% { transform: scale(2.5); opacity: 0;  }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #080c12; }
        ::-webkit-scrollbar-thumb { background: #2d3748; }
      `}</style>

      {/* Header */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "center",
        marginBottom: 24, paddingBottom: 16, borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 3, fontFamily: "'Courier New', monospace", marginBottom: 4 }}>
            AQUA INTELLIGENCE PLATFORM
          </div>
          <h1 style={{ margin: 0, fontSize: 30, fontWeight: 700, letterSpacing: -0.5, color: "#f7fafc" }}>
            Water Quality Monitor
          </h1>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ fontSize: 12, color: "#4a5568" }}>
            {lastUpdate ? `Updated ${lastUpdate}` : "Waiting for data…"}
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: 8, padding: "6px 14px", borderRadius: 20,
            background: connected ? "rgba(0,229,160,0.08)" : "rgba(255,59,92,0.08)",
            border: `1px solid ${connected ? "rgba(0,229,160,0.2)" : "rgba(255,59,92,0.2)"}`,
          }}>
            <PulseRing color={connected ? "#00e5a0" : "#ff3b5c"} />
            <span style={{ fontSize: 11, color: connected ? "#00e5a0" : "#ff3b5c", fontFamily: "'Courier New', monospace", letterSpacing: 1 }}>
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      {/* Row 1: Gauge + Sensors */}
      <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 16, marginBottom: 16 }}>
        <div style={{
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 10, padding: "16px 12px", display: "flex", flexDirection: "column", alignItems: "center",
        }}>
          <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 2, fontFamily: "'Courier New', monospace", marginBottom: 2 }}>CURRENT WQI</div>
          <div style={{ fontSize: 11, color: "#6b7a8d", marginBottom: 4 }}>Random Forest</div>
          <WQIGauge value={rf.wqi} />
          <div style={{
            fontSize: 13, fontWeight: 600, padding: "4px 18px", borderRadius: 20,
            background: `${rfColor}18`, color: rfColor, border: `1px solid ${rfColor}40`, marginTop: 4,
          }}>
            {rf.status ?? "—"}
          </div>
          <div style={{ fontSize: 10, color: "#4a5568", marginTop: 8 }}>threshold ≥ 50 = Safe</div>
        </div>

        <div style={{
          background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 10, padding: "20px 28px",
        }}>
          <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 2, fontFamily: "'Courier New', monospace", marginBottom: 18 }}>
            LIVE SENSOR READINGS — {latest?.node_id ?? "NODE-001"}
          </div>
          <SensorBar label="pH"          value={sensor.pH}             min={5}  max={9}   unit=""      color="#60a5fa" />
          <SensorBar label="TURBIDITY"   value={sensor.Turbidity_NTU}  min={0}  max={10}  unit="NTU"   color="#a78bfa" />
          <SensorBar label="TDS"         value={sensor.TDS_ppm}        min={0}  max={600} unit="ppm"   color="#34d399" />
          <SensorBar label="TEMPERATURE" value={sensor.Temperature_C}  min={10} max={40}  unit="°C"    color="#fbbf24" />
          <SensorBar label="FLOW RATE"   value={sensor.Flow_Rate_Lmin} min={0}  max={5}   unit="L/min" color="#f87171" />
        </div>
      </div>

      {/* Row 2: LSTM Forecasts */}
      <div style={{
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 10, padding: "18px 28px", marginBottom: 16,
      }}>
        <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 2, fontFamily: "'Courier New', monospace", marginBottom: 16 }}>
          LSTM MULTI-HORIZON FORECAST
        </div>
        {lstm.ready === false ? (
          <div style={{ color: "#4a5568", fontSize: 13 }}>
            ⏳ Warming up… {lstm.window_size ?? 0} / {lstm.required ?? 8} readings collected
            <div style={{ height: 4, background: "#1e2633", borderRadius: 2, marginTop: 8, overflow: "hidden", maxWidth: 300 }}>
              <div style={{
                height: "100%",
                width: `${((lstm.window_size ?? 0) / (lstm.required ?? 8)) * 100}%`,
                background: "#60a5fa", borderRadius: 2, transition: "width 0.4s",
              }} />
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
            <StatCard label="+15 MIN"    value={lstm.wqi_15min?.toFixed(1)} unit="WQI" color={wqiColor(lstm.wqi_15min)} sub={wqiLabel(lstm.wqi_15min)} />
            <StatCard label="+30 MIN"    value={lstm.wqi_30min?.toFixed(1)} unit="WQI" color={wqiColor(lstm.wqi_30min)} sub={wqiLabel(lstm.wqi_30min)} />
            <StatCard label="+60 MIN"    value={lstm.wqi_60min?.toFixed(1)} unit="WQI" color={wqiColor(lstm.wqi_60min)} sub={wqiLabel(lstm.wqi_60min)} />
            <StatCard label="CURRENT RF" value={rf.wqi?.toFixed(1)}         unit="WQI" color={rfColor}                  sub={rf.status} />
          </div>
        )}
      </div>

      {/* Row 3: WQI Timeline */}
      <div style={{
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 10, padding: "18px 28px", marginBottom: 16,
      }}>
        <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 2, fontFamily: "'Courier New', monospace", marginBottom: 16 }}>
          WQI TIMELINE — RF vs LSTM FORECASTS vs GROUND TRUTH
        </div>
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={readings} margin={{ top: 4, right: 16, bottom: 0, left: -10 }}>
            <CartesianGrid stroke="#1a2233" strokeDasharray="3 3" />
            <XAxis dataKey="time" tick={{ fill: "#4a5568", fontSize: 10 }} interval="preserveStartEnd" />
            <YAxis domain={[30, 100]} tick={{ fill: "#4a5568", fontSize: 10 }} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={50} stroke="#ff3b5c" strokeDasharray="4 4" strokeOpacity={0.5} />
            <Line type="monotone" dataKey="rfWqi"     name="RF (current)" stroke="#00e5a0" strokeWidth={2}   dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="lstmWqi15" name="+15min"        stroke="#60a5fa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
            <Line type="monotone" dataKey="lstmWqi30" name="+30min"        stroke="#a78bfa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
            <Line type="monotone" dataKey="lstmWqi60" name="+60min"        stroke="#f59e0b" strokeWidth={1.5} dot={false} strokeDasharray="4 2" isAnimationActive={false} />
            <Line type="monotone" dataKey="gtWqi"     name="Ground Truth"  stroke="#ff3b5c" strokeWidth={1}   dot={false} strokeOpacity={0.4}  isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
        <div style={{ display: "flex", gap: 24, marginTop: 12, flexWrap: "wrap" }}>
          {[
            { color: "#00e5a0", label: "RF (current)" },
            { color: "#60a5fa", label: "LSTM +15min"  },
            { color: "#a78bfa", label: "LSTM +30min"  },
            { color: "#f59e0b", label: "LSTM +60min"  },
            { color: "#ff3b5c", label: "Ground Truth" },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 22, height: 2, background: color }} />
              <span style={{ fontSize: 11, color: "#6b7a8d" }}>{label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Row 4: Sensor Charts */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 16, marginBottom: 24 }}>
        {[
          { key: "pH",        label: "PH",               color: "#60a5fa", domain: [5, 9]   },
          { key: "turbidity", label: "TURBIDITY (NTU)",  color: "#a78bfa", domain: [0, 10]  },
          { key: "tds",       label: "TDS (PPM)",        color: "#34d399", domain: [0, 600] },
          { key: "temp",      label: "TEMPERATURE (°C)", color: "#fbbf24", domain: [10, 40] },
        ].map(({ key, label, color, domain }) => (
          <div key={key} style={{
            background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: 10, padding: "16px 20px",
          }}>
            <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 2, fontFamily: "'Courier New', monospace", marginBottom: 12 }}>
              {label}
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <AreaChart data={readings} margin={{ top: 2, right: 4, bottom: 0, left: -20 }}>
                <defs>
                  <linearGradient id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={color} stopOpacity={0.25} />
                    <stop offset="95%" stopColor={color} stopOpacity={0}    />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1a2233" strokeDasharray="3 3" />
                <XAxis dataKey="time" hide />
                <YAxis domain={domain} tick={{ fill: "#4a5568", fontSize: 9 }} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey={key} name={label} stroke={color}
                  fill={`url(#grad-${key})`} strokeWidth={1.5} dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div style={{
        textAlign: "center", fontSize: 11, color: "#2d3748",
        paddingTop: 14, borderTop: "1px solid rgba(255,255,255,0.04)",
      }}>
        Water Quality Predictor · RF + LSTM Multi-Horizon · WebSocket live feed
      </div>
    </div>
  );
}