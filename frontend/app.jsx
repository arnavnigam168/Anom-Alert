import { useMemo, useState } from 'react'
import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import { Line, Bar, ResponsiveContainer, LineChart, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts'

const timeSeries = [
  { time: '08:00', pH: 7.2, temperature: 22.5, oxygen: 8.9 },
  { time: '10:00', pH: 7.4, temperature: 23.1, oxygen: 8.7 },
  { time: '12:00', pH: 7.1, temperature: 23.9, oxygen: 8.3 },
  { time: '14:00', pH: 6.9, temperature: 24.6, oxygen: 8.1 },
  { time: '16:00', pH: 7.0, temperature: 24.3, oxygen: 8.5 },
  { time: '18:00', pH: 7.3, temperature: 23.8, oxygen: 8.8 }
]

const statusStyles = {
  Normal: 'bg-emerald-500/15 text-emerald-300',
  Stable: 'bg-sky-500/15 text-sky-300',
  Review: 'bg-amber-500/15 text-amber-300'
}

const verdictStyles = {
  PASS: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/40',
  REVIEW: 'bg-amber-500/10 text-amber-300 border-amber-500/40',
  FAIL: 'bg-rose-500/10 text-rose-300 border-rose-500/40'
}

const initialInput = {
  pH_mean: 6.95,
  temp_max: 37.9,
  oxygen_variance: 0.16,
  ts_mean: 1.03,
  ts_std: 0.17,
  ts_max: 1.31,
  ts_min: 0.77,
  ts_slope: 0.01,
  ts_max_jump: 0.12,
  ts_rolling_var_mean: 0.01,
  ts_spike_count: 0
}

const defaultPrediction = {
  prediction: 'REVIEW',
  confidence: 0,
  top_features: [],
  explanation: 'Awaiting model inference.'
}

const fieldMeta = [
  { key: 'pH_mean', label: 'pH mean', range: '6.8 - 7.4' },
  { key: 'temp_max', label: 'Temperature max', range: '22.0 - 40.0' },
  { key: 'oxygen_variance', label: 'Oxygen variance', range: '0.02 - 0.90' },
  { key: 'ts_mean', label: 'TS mean', range: '0.7 - 1.4' },
  { key: 'ts_std', label: 'TS std', range: '0.02 - 0.35' },
  { key: 'ts_max', label: 'TS max', range: '0.8 - 2.0' },
  { key: 'ts_min', label: 'TS min', range: '0.2 - 1.4' },
  { key: 'ts_slope', label: 'TS slope', range: '-0.2 - 0.2' },
  { key: 'ts_max_jump', label: 'TS max jump', range: '0.0 - 1.2' },
  { key: 'ts_rolling_var_mean', label: 'TS rolling var', range: '0.0 - 0.2' },
  { key: 'ts_spike_count', label: 'TS spike count', range: '0 - 8' }
]

const navItems = [
  { label: 'Dashboard', to: '/' },
  { label: 'Batch History', to: '/history' },
  { label: 'Audit Logs', to: '/logs' }
]

function formatTimestamp(date = new Date()) {
  return date.toISOString().slice(0, 19).replace('T', ' ')
}

function buildBatchId(index) {
  const now = new Date()
  const day = String(now.getDate()).padStart(2, '0')
  const month = String(now.getMonth() + 1).padStart(2, '0')
  return `BCH-${month}${day}-${String(index).padStart(2, '0')}`
}

function Sidebar({ currentBatchId, predictionResult }) {
  return (
    <aside className="space-y-3 rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
      <div className="inline-flex items-center gap-3 rounded-3xl bg-slate-800/80 px-4 py-3 text-slate-100 shadow-sm shadow-slate-900/40">
        <span className="inline-flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 to-sky-500 text-slate-950 font-semibold">A</span>
        <div>
          <h1 className="text-base font-semibold">AnomAlert Bio</h1>
          <p className="text-xs text-slate-400">AI monitoring & audit center</p>
        </div>
      </div>

      <div className="space-y-2 rounded-3xl border border-slate-800/70 bg-slate-950/80 p-3">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Navigation</p>
        <nav className="space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex w-full items-center justify-between rounded-2xl px-3 py-2 text-left text-sm font-medium transition ${
                  isActive ? 'bg-slate-800 text-white shadow-lg shadow-slate-950/20' : 'text-slate-400 hover:bg-slate-800/80 hover:text-slate-100'
                }`
              }
            >
              {({ isActive }) => (
                <>
                  <span>{item.label}</span>
                  {isActive ? (
                    <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] text-slate-300">Live</span>
                  ) : null}
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="rounded-3xl border border-slate-800/70 bg-slate-950/80 p-3">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Summary</p>
        <div className="mt-3 space-y-2">
          <div className="rounded-3xl bg-slate-900/70 p-3">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Current Batch</p>
            <h2 className="mt-1 text-base font-semibold text-white">{currentBatchId}</h2>
          </div>
          <div className="rounded-3xl bg-slate-900/70 p-3">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">AI Verdict</p>
            <div className={`mt-2 inline-flex rounded-full border px-3 py-1 text-xs font-semibold ${verdictStyles[predictionResult.prediction] ?? verdictStyles.REVIEW}`}>
              {predictionResult.prediction}
            </div>
          </div>
          <div className="rounded-3xl bg-slate-900/70 p-3">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Confidence</p>
            <p className="mt-1 text-lg font-semibold text-white">{(predictionResult.confidence * 100).toFixed(1)}%</p>
          </div>
        </div>
      </div>
    </aside>
  )
}

function DashboardPage({
  predictionResult,
  loading,
  error,
  inputValues,
  updateInputValue,
  runPrediction
}) {
  const shapData = useMemo(() => {
    if (!predictionResult.top_features?.length) {
      return [
        { feature: 'Feature 1', importance: 0 },
        { feature: 'Feature 2', importance: 0 },
        { feature: 'Feature 3', importance: 0 }
      ]
    }
    return predictionResult.top_features.map((feature, idx) => ({
      feature,
      importance: Math.max(100 - idx * 20, 40)
    }))
  }, [predictionResult.top_features])

  const parameterRows = useMemo(() => {
    return fieldMeta.map((item) => {
      const value = Number(inputValues[item.key] ?? 0)
      let status = 'Normal'
      if (Number.isNaN(value)) {
        status = 'Review'
      } else if (item.key === 'temp_max' && value > 38.5) {
        status = 'Review'
      } else if (item.key === 'oxygen_variance' && value > 0.25) {
        status = 'Review'
      } else if (item.key === 'ts_spike_count' && value > 1) {
        status = 'Stable'
      } else if (item.key === 'ts_std' && value > 0.2) {
        status = 'Stable'
      }
      return {
        parameter: item.label,
        key: item.key,
        range: item.range,
        status
      }
    })
  }, [inputValues])

  return (
    <main className="space-y-3">
      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">Dashboard</p>
            <h2 className="mt-2 text-lg font-semibold text-white">Live prediction panel</h2>
            <p className="mt-2 text-sm leading-6 text-slate-400">Run model inference and inspect top contributing features in real time.</p>
            <p className="mt-2 text-sm text-slate-300">Top features: {predictionResult.top_features.join(', ') || 'N/A'}</p>
            <p className="mt-1 text-sm text-slate-400">{predictionResult.explanation}</p>
            {error ? <p className="mt-2 text-sm text-rose-300">Error: {error}</p> : null}
          </div>
          <div className="inline-flex items-center gap-3 rounded-3xl bg-slate-950/70 px-4 py-2 text-sm text-slate-300 shadow-inner shadow-slate-950/20">
            {loading ? (
              <>
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-cyan-300 border-t-transparent"></span>
                Requesting prediction...
              </>
            ) : (
              <>
                <span className="h-2.5 w-2.5 rounded-full bg-emerald-400"></span> Live sensor sync enabled
              </>
            )}
          </div>
        </div>
      </section>

      <section className="grid gap-3 xl:grid-cols-[1.5fr,1fr]">
        <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Time Series</p>
              <h3 className="mt-1 text-base font-semibold text-white">pH, temperature and oxygen</h3>
            </div>
            <div className="rounded-2xl bg-slate-950/80 px-3 py-1 text-xs text-slate-300">Last 12 hours</div>
          </div>
          <div className="mt-4 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timeSeries} margin={{ top: 10, right: 12, left: -24, bottom: 4 }}>
                <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" />
                <XAxis dataKey="time" tick={{ fill: '#94A3B8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#94A3B8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16 }} itemStyle={{ color: '#fff' }} labelStyle={{ color: '#94A3B8' }} />
                <Legend wrapperStyle={{ color: '#94A3B8', fontSize: 12 }} />
                <Line type="monotone" dataKey="pH" stroke="#38bdf8" strokeWidth={2.5} dot={{ r: 3, fill: '#38bdf8' }} />
                <Line type="monotone" dataKey="temperature" stroke="#0ea5e9" strokeWidth={2.5} dot={{ r: 3, fill: '#0ea5e9' }} />
                <Line type="monotone" dataKey="oxygen" stroke="#22c55e" strokeWidth={2.5} dot={{ r: 3, fill: '#22c55e' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Feature Importance</p>
              <h3 className="mt-1 text-base font-semibold text-white">SHAP view</h3>
            </div>
            <div className="rounded-2xl bg-slate-950/80 px-3 py-1 text-xs text-slate-300">Explainability</div>
          </div>
          <div className="mt-4 h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapData} margin={{ top: 10, right: 8, left: -14, bottom: 4 }}>
                <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" />
                <XAxis dataKey="feature" tick={{ fill: '#94A3B8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#94A3B8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16 }} itemStyle={{ color: '#fff' }} labelStyle={{ color: '#94A3B8' }} />
                <Bar dataKey="importance" radius={[10, 10, 0, 0]} fill="#38bdf8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <div className="flex items-center justify-between gap-3">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Parameters</p>
            <h3 className="mt-1 text-base font-semibold text-white">Live sample status</h3>
          </div>
          <button
            onClick={runPrediction}
            disabled={loading}
            className="rounded-2xl border border-slate-700/80 bg-slate-950/80 px-4 py-2 text-sm text-slate-300 transition hover:border-slate-600 hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? 'Predicting...' : 'Run prediction'}
          </button>
        </div>
        <div className="mt-4 overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/70">
          <table className="min-w-full text-sm text-slate-200">
            <thead className="bg-slate-900/90 text-slate-400">
              <tr>
                <th className="px-4 py-3 text-left uppercase tracking-[0.12em]">Parameter</th>
                <th className="px-4 py-3 text-left uppercase tracking-[0.12em]">Value</th>
                <th className="px-4 py-3 text-left uppercase tracking-[0.12em]">Range</th>
                <th className="px-4 py-3 text-left uppercase tracking-[0.12em]">Status</th>
              </tr>
            </thead>
            <tbody>
              {parameterRows.map((item) => (
                <tr key={item.key} className="border-t border-slate-800/80 transition hover:bg-slate-900/70">
                  <td className="px-4 py-3 font-medium text-slate-100">{item.parameter}</td>
                  <td className="px-4 py-3 text-slate-300">
                    <input
                      type="number"
                      step="any"
                      value={inputValues[item.key]}
                      onChange={(e) => updateInputValue(item.key, e.target.value)}
                      className="w-24 rounded-lg border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100 outline-none focus:border-cyan-400"
                    />
                  </td>
                  <td className="px-4 py-3 text-slate-400">{item.range}</td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.16em] ${statusStyles[item.status] ?? statusStyles.Review}`}>
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  )
}

function BatchHistoryPage({ batchHistory, latestBatch }) {
  return (
    <main className="space-y-3">
      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">Batch History</p>
        <h2 className="mt-2 text-lg font-semibold text-white">Past predictions</h2>
        {latestBatch ? (
          <div className="mt-3 rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
            <p className="text-sm font-semibold text-white">Latest batch: {latestBatch.id}</p>
            <p className="text-xs text-slate-400">{latestBatch.time}</p>
            <p className="mt-1 text-sm text-slate-300">
              {latestBatch.prediction} ({(latestBatch.confidence * 100).toFixed(1)}%) - {latestBatch.topFeatures.join(', ') || 'N/A'}
            </p>
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-400">No predictions yet. Run prediction from Dashboard.</p>
        )}
      </section>

      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <h3 className="text-base font-semibold text-white">All batches</h3>
        <div className="mt-3 max-h-[540px] space-y-2 overflow-y-auto pr-1">
          {batchHistory.length === 0 ? (
            <p className="text-sm text-slate-400">History will appear here after predictions.</p>
          ) : (
            batchHistory.map((entry) => (
              <div key={`${entry.id}-${entry.time}`} className="rounded-2xl border border-slate-800 bg-slate-950/70 p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-white">{entry.id}</p>
                  <span className={`rounded-full border px-2 py-0.5 text-xs ${verdictStyles[entry.prediction] ?? verdictStyles.REVIEW}`}>{entry.prediction}</span>
                </div>
                <p className="mt-1 text-xs text-slate-400">{entry.time}</p>
                <p className="mt-1 text-sm text-slate-300">Confidence: {(entry.confidence * 100).toFixed(1)}%</p>
                <p className="mt-1 text-xs text-slate-500">Top features: {entry.topFeatures.join(', ') || 'N/A'}</p>
              </div>
            ))
          )}
        </div>
      </section>
    </main>
  )
}

function AuditLogsPage({ auditLogs }) {
  return (
    <main className="space-y-3">
      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <p className="text-xs uppercase tracking-[0.24em] text-cyan-300/70">Audit Logs</p>
        <h2 className="mt-2 text-lg font-semibold text-white">System and prediction actions</h2>
        <p className="mt-2 text-sm text-slate-400">Logs include timestamps, actions, and payload snapshots for demo traceability.</p>
      </section>

      <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-4 shadow-soft shadow-black/20 backdrop-blur-lg">
        <div className="max-h-[640px] space-y-2 overflow-y-auto pr-1">
          {auditLogs.length === 0 ? (
            <p className="text-sm text-slate-400">No logs available yet.</p>
          ) : (
            auditLogs.map((entry) => (
              <div key={`${entry.time}-${entry.action}`} className="rounded-2xl border border-slate-800/80 bg-slate-950/70 p-3">
                <div className="flex items-center justify-between gap-2 text-xs text-slate-400">
                  <span>{entry.time}</span>
                  <span className="rounded-full bg-slate-800/80 px-2 py-1 uppercase tracking-[0.14em] text-slate-300">System</span>
                </div>
                <p className="mt-2 text-sm leading-6 text-slate-200">{entry.action}</p>
              </div>
            ))
          )}
        </div>
      </section>
    </main>
  )
}

function AppShell() {
  const [inputValues, setInputValues] = useState(initialInput)
  const [predictionResult, setPredictionResult] = useState(defaultPrediction)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [batchHistory, setBatchHistory] = useState([])
  const [auditLogs, setAuditLogs] = useState([
    { time: formatTimestamp(), action: 'Dashboard initialized and awaiting prediction.' }
  ])
  const [batchCount, setBatchCount] = useState(1)

  const currentBatchId = useMemo(() => buildBatchId(batchCount), [batchCount])
  const latestBatch = batchHistory[0]

  function updateInputValue(key, rawValue) {
    const numericValue = rawValue === '' ? 0 : Number(rawValue)
    setInputValues((prev) => ({
      ...prev,
      [key]: Number.isFinite(numericValue) ? numericValue : 0
    }))
  }

  async function runPrediction() {
    setLoading(true)
    setError('')
    const timestamp = formatTimestamp()

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputValues)
      })

      if (!response.ok) {
        throw new Error(`Backend error (${response.status})`)
      }

      const data = await response.json()
      const cleanResult = {
        prediction: data?.prediction ?? 'REVIEW',
        confidence: Number(data?.confidence ?? 0),
        top_features: Array.isArray(data?.top_features) ? data.top_features : [],
        explanation: data?.explanation ?? 'No explanation available.'
      }

      setPredictionResult(cleanResult)
      setBatchHistory((prev) => [
        {
          id: currentBatchId,
          time: timestamp,
          prediction: cleanResult.prediction,
          confidence: cleanResult.confidence,
          topFeatures: cleanResult.top_features
        },
        ...prev
      ])
      setAuditLogs((prev) => [
        {
          time: timestamp,
          action: `Prediction completed for ${currentBatchId}: ${cleanResult.prediction} (${(cleanResult.confidence * 100).toFixed(1)}%).`
        },
        {
          time: timestamp,
          action: `Input payload: ${JSON.stringify(inputValues)}`
        },
        ...prev
      ])
      setBatchCount((v) => v + 1)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Prediction failed.'
      setError(message)
      setAuditLogs((prev) => [
        {
          time: timestamp,
          action: `Prediction API unavailable. Logged simulated audit event for ${currentBatchId}.`
        },
        {
          time: timestamp,
          action: `Input payload (simulated): ${JSON.stringify(inputValues)}`
        },
        ...prev
      ])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-[1200px] px-4 py-4">
        <div className="grid gap-3 xl:grid-cols-[260px,1fr]">
          <Sidebar currentBatchId={currentBatchId} predictionResult={predictionResult} />

          <Routes>
            <Route
              path="/"
              element={
                <DashboardPage
                  predictionResult={predictionResult}
                  loading={loading}
                  error={error}
                  inputValues={inputValues}
                  updateInputValue={updateInputValue}
                  runPrediction={runPrediction}
                />
              }
            />
            <Route path="/history" element={<BatchHistoryPage batchHistory={batchHistory} latestBatch={latestBatch} />} />
            <Route path="/logs" element={<AuditLogsPage auditLogs={auditLogs} />} />
          </Routes>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}

