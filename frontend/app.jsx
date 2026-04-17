import { Line, Bar, ResponsiveContainer, LineChart, BarChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts'

const batchOverview = {
  id: 'BCH-0426-19',
  verdict: 'REVIEW',
  confidence: 82
}

const timeSeries = [
  { time: '08:00', pH: 7.2, temperature: 22.5, oxygen: 8.9 },
  { time: '10:00', pH: 7.4, temperature: 23.1, oxygen: 8.7 },
  { time: '12:00', pH: 7.1, temperature: 23.9, oxygen: 8.3 },
  { time: '14:00', pH: 6.9, temperature: 24.6, oxygen: 8.1 },
  { time: '16:00', pH: 7.0, temperature: 24.3, oxygen: 8.5 },
  { time: '18:00', pH: 7.3, temperature: 23.8, oxygen: 8.8 }
]

const shapData = [
  { feature: 'Salinity', importance: 92 },
  { feature: 'Temperature', importance: 78 },
  { feature: 'pH', importance: 65 },
  { feature: 'Turbidity', importance: 52 },
  { feature: 'Nitrate', importance: 44 }
]

const parameters = [
  { parameter: 'pH', value: '7.2', range: '6.8 - 7.4', status: 'Normal' },
  { parameter: 'Temperature', value: '23.9°C', range: '22.0 - 24.5°C', status: 'Normal' },
  { parameter: 'Oxygen', value: '8.5 mg/L', range: '8.0 - 9.5 mg/L', status: 'Stable' },
  { parameter: 'Salinity', value: '34 PSU', range: '33 - 35 PSU', status: 'Review' }
]

const auditLogs = [
  { time: '2026-04-16 18:28', action: 'AI model flagged batch for manual review' },
  { time: '2026-04-16 18:12', action: 'Batch parameters synchronized from sensor cluster' },
  { time: '2026-04-16 17:55', action: 'New SHAP features added to explanation model' },
  { time: '2026-04-16 17:22', action: 'Audit record created for batch BCH-0426-19' }
]

const navItems = [
  { label: 'Dashboard', active: true },
  { label: 'Batch History', active: false },
  { label: 'Audit Logs', active: false }
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

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-[1760px] px-4 py-6 sm:px-6 lg:px-8">
        <div className="grid gap-6 xl:grid-cols-[280px,1fr]">
          <aside className="space-y-6 rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-3 rounded-3xl bg-slate-800/80 px-4 py-3 text-slate-100 shadow-sm shadow-slate-900/40">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br from-cyan-400 to-sky-500 text-slate-950 font-semibold">A</span>
                <div>
                  <h1 className="text-lg font-semibold">AnomAlert Bio</h1>
                  <p className="text-sm text-slate-400">AI monitoring & audit center</p>
                </div>
              </div>
            </div>

            <div className="space-y-3 rounded-3xl border border-slate-800/70 bg-slate-950/80 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Navigation</p>
              <nav className="space-y-2">
                {navItems.map((item) => (
                  <button
                    key={item.label}
                    className={`flex w-full items-center justify-between rounded-2xl px-4 py-3 text-left text-sm font-medium transition ${
                      item.active
                        ? 'bg-slate-800 text-white shadow-lg shadow-slate-950/20'
                        : 'text-slate-400 hover:bg-slate-800/80 hover:text-slate-100'
                    }`}
                  >
                    <span>{item.label}</span>
                    {item.active ? <span className="rounded-full bg-slate-700 px-2 py-0.5 text-[11px] uppercase tracking-[0.18em] text-slate-300">Live</span> : null}
                  </button>
                ))}
              </nav>
            </div>

            <div className="rounded-3xl border border-slate-800/70 bg-slate-950/80 p-4">
              <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Summary</p>
              <div className="mt-4 space-y-3">
                <div className="rounded-3xl bg-slate-900/70 p-4">
                  <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Current Batch</p>
                  <h2 className="mt-2 text-xl font-semibold text-white">{batchOverview.id}</h2>
                </div>
                <div className="rounded-3xl bg-slate-900/70 p-4">
                  <p className="text-xs uppercase tracking-[0.24em] text-slate-500">AI verdict</p>
                  <div className={`mt-3 inline-flex rounded-full border px-3 py-1 text-sm font-semibold ${verdictStyles[batchOverview.verdict]}`}>
                    {batchOverview.verdict}
                  </div>
                </div>
                <div className="rounded-3xl bg-slate-900/70 p-4">
                  <p className="text-xs uppercase tracking-[0.24em] text-slate-500">Confidence</p>
                  <p className="mt-2 text-3xl font-semibold text-white">{batchOverview.confidence}%</p>
                </div>
              </div>
            </div>
          </aside>

          <main className="space-y-6">
            <section className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
              <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.28em] text-cyan-300/70">Overview</p>
                  <h2 className="mt-3 text-3xl font-semibold text-white">Batch dashboard</h2>
                  <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                    Monitor the latest batch, verify model outputs, and review audit events from one unified interface.
                  </p>
                </div>
                <div className="inline-flex items-center gap-3 rounded-3xl bg-slate-950/70 px-4 py-3 text-sm text-slate-300 shadow-inner shadow-slate-950/20">
                  <span className="h-2.5 w-2.5 rounded-full bg-emerald-400"></span> Live sensor sync enabled
                </div>
              </div>
            </section>

            <section className="grid gap-6 xl:grid-cols-[1.5fr,1fr]">
              <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Time series</p>
                    <h3 className="mt-2 text-xl font-semibold text-white">pH, temperature & oxygen</h3>
                  </div>
                  <div className="rounded-2xl bg-slate-950/80 px-3 py-2 text-sm text-slate-300">Last 12 hours</div>
                </div>
                <div className="mt-6 h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={timeSeries} margin={{ top: 10, right: 12, left: -24, bottom: 4 }}>
                      <CartesianGrid stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" />
                      <XAxis dataKey="time" tick={{ fill: '#94A3B8', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#94A3B8', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16 }} itemStyle={{ color: '#fff' }} labelStyle={{ color: '#94A3B8' }} />
                      <Legend wrapperStyle={{ color: '#94A3B8', fontSize: 13 }} />
                      <Line type="monotone" dataKey="pH" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4, fill: '#38bdf8' }} />
                      <Line type="monotone" dataKey="temperature" stroke="#0ea5e9" strokeWidth={3} dot={{ r: 4, fill: '#0ea5e9' }} />
                      <Line type="monotone" dataKey="oxygen" stroke="#22c55e" strokeWidth={3} dot={{ r: 4, fill: '#22c55e' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Feature importance</p>
                    <h3 className="mt-2 text-xl font-semibold text-white">SHAP analysis</h3>
                  </div>
                  <div className="rounded-2xl bg-slate-950/80 px-3 py-2 text-sm text-slate-300">Model explainability</div>
                </div>
                <div className="mt-6 h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData} margin={{ top: 10, right: 8, left: -14, bottom: 4 }}>
                      <CartesianGrid vertical={false} stroke="rgba(148,163,184,0.12)" strokeDasharray="4 4" />
                      <XAxis dataKey="feature" tick={{ fill: '#94A3B8', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: '#94A3B8', fontSize: 12 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16 }} itemStyle={{ color: '#fff' }} labelStyle={{ color: '#94A3B8' }} />
                      <Bar dataKey="importance" radius={[12, 12, 0, 0]} fill="#38bdf8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>

            <section className="grid gap-6 xl:grid-cols-[1.4fr,1fr]">
              <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Parameters</p>
                    <h3 className="mt-2 text-xl font-semibold text-white">Live sample status</h3>
                  </div>
                  <button className="rounded-2xl border border-slate-700/80 bg-slate-950/80 px-4 py-2 text-sm text-slate-300 transition hover:border-slate-600 hover:text-white">
                    Refresh data
                  </button>
                </div>
                <div className="mt-6 overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/70">
                  <table className="min-w-full text-sm text-slate-200">
                    <thead className="bg-slate-900/90 text-slate-400">
                      <tr>
                        <th className="px-5 py-4 text-left uppercase tracking-[0.16em]">Parameter</th>
                        <th className="px-5 py-4 text-left uppercase tracking-[0.16em]">Value</th>
                        <th className="px-5 py-4 text-left uppercase tracking-[0.16em]">Range</th>
                        <th className="px-5 py-4 text-left uppercase tracking-[0.16em]">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {parameters.map((item) => (
                        <tr key={item.parameter} className="border-t border-slate-800/80 transition hover:bg-slate-900/70">
                          <td className="px-5 py-4 font-medium text-slate-100">{item.parameter}</td>
                          <td className="px-5 py-4 text-slate-300">{item.value}</td>
                          <td className="px-5 py-4 text-slate-400">{item.range}</td>
                          <td className="px-5 py-4">
                            <span className={`inline-flex rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] ${statusStyles[item.status]}`}>
                              {item.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-3xl border border-slate-800/80 bg-slate-900/80 p-6 shadow-soft shadow-black/20 backdrop-blur-lg">
                <div>
                  <p className="text-sm uppercase tracking-[0.24em] text-slate-500">Audit log</p>
                  <h3 className="mt-2 text-xl font-semibold text-white">Recent actions</h3>
                </div>
                <div className="mt-6 space-y-4">
                  {auditLogs.map((entry) => (
                    <div key={entry.time} className="rounded-3xl border border-slate-800/80 bg-slate-950/70 p-4 transition hover:border-slate-700/70 hover:bg-slate-900/80">
                      <div className="flex items-center justify-between gap-3 text-sm text-slate-400">
                        <span>{entry.time}</span>
                        <span className="rounded-full bg-slate-800/80 px-3 py-1 text-[11px] uppercase tracking-[0.18em] text-slate-300">System</span>
                      </div>
                      <p className="mt-3 text-sm leading-6 text-slate-200">{entry.action}</p>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  )
}

export default App
