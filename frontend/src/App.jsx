import { useState, useRef, useCallback } from 'react'
import TickerInput from './components/TickerInput.jsx'
import AgentTraceView from './components/AgentTraceView.jsx'
import MemoPanel from './components/MemoPanel.jsx'
import RatioTable from './components/RatioTable.jsx'
import SentimentGauge from './components/SentimentGauge.jsx'
import RiskFlagList from './components/RiskFlagList.jsx'
import AgentRoster from './components/AgentRoster.jsx'
import SystemStatus from './components/SystemStatus.jsx'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [ticker, setTicker]             = useState('')
  const [running, setRunning]           = useState(false)
  const [error, setError]               = useState(null)
  const [traceEntries, setTraceEntries] = useState([])
  const [memo, setMemo]                 = useState(null)
  const [ratios, setRatios]             = useState(null)
  const [sentiment, setSentiment]       = useState(null)
  const [riskFlags, setRiskFlags]       = useState(null)
  const [citationCoverage, setCoverage] = useState(null)
  const traceRef = useRef(null)

  const resetState = () => {
    setError(null)
    setTraceEntries([])
    setMemo(null)
    setRatios(null)
    setSentiment(null)
    setRiskFlags(null)
    setCoverage(null)
  }

  const handleSubmit = useCallback(async (explicitTicker) => {
    const t = (explicitTicker || ticker).trim().toUpperCase()
    if (!t) return
    setTicker(t)
    resetState()
    setRunning(true)

    try {
      const url = `${API}/diligence/stream?ticker=${encodeURIComponent(t)}`
      const res = await fetch(url, { headers: { Accept: 'text/event-stream' } })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `HTTP ${res.status}`)
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6).trim()
          if (!raw) continue
          let evt
          try { evt = JSON.parse(raw) } catch { continue }

          if (evt.type === 'trace_entry') {
            setTraceEntries((prev) => [...prev, evt])
          } else if (evt.type === 'state_update') {
            if (evt.key === 'memo')              setMemo(evt.value)
            if (evt.key === 'ratios')            setRatios(evt.value)
            if (evt.key === 'sentiment')         setSentiment(evt.value)
            if (evt.key === 'risk_flags')        setRiskFlags(evt.value)
            if (evt.key === 'citation_coverage') setCoverage(evt.value)
          } else if (evt.type === 'end') {
            break
          }
        }
      }
    } catch (e) {
      setError(e.message || 'Unknown error')
    } finally {
      setRunning(false)
    }
  }, [ticker])

  const handleClaimClick = useCallback((agentName) => {
    traceRef.current?.scrollToNode(agentName)
  }, [])

  // Drives visibility of idle-only sections
  const isIdle = !traceEntries.length && !running

  return (
    <div className="min-h-screen bg-ink text-ink-text">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="bg-surface border-b border-line px-6 py-4 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <h1 className="font-display text-xl font-semibold text-ink-text tracking-tight">
            Financial Due-Diligence
            <span className="ml-3 font-body text-sm font-normal text-muted">
              multi-agent analysis
            </span>
          </h1>
          {/* System status lives in the header, hidden when results are active */}
          {isIdle && <SystemStatus />}
        </div>
      </header>

      {/* ── Main ───────────────────────────────────────────────────── */}
      <main className="max-w-7xl mx-auto px-6 py-10 space-y-6">

        {/* Command bar — always visible */}
        <section className="bg-surface rounded-xl border border-line p-6 shadow-sm">
          <TickerInput
            value={ticker}
            onChange={setTicker}
            onSubmit={handleSubmit}
            disabled={running}
          />
          {error && (
            <div className="mt-4 flex items-start gap-2 rounded-lg border-l-2 border-rose bg-rose/10 px-4 py-2.5 text-sm text-rose font-body">
              <span className="mt-px">⚠</span>
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Idle-only: agent architecture strip */}
        {isIdle && <AgentRoster />}

        {/* Results grid — shown once streaming begins */}
        {!isIdle && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
            {/* Left column: pipeline trace + data panels */}
            <div className="lg:col-span-1 space-y-4">
              <AgentTraceView ref={traceRef} entries={traceEntries} />
              {ratios    && <RatioTable ratios={ratios} />}
              {sentiment && <SentimentGauge sentiment={sentiment} />}
              {riskFlags != null && <RiskFlagList flags={riskFlags} />}
            </div>

            {/* Right column: investment memo */}
            <div className="lg:col-span-2">
              {memo ? (
                <div className="bg-surface rounded-xl border border-line p-6">
                  <MemoPanel
                    memo={memo}
                    citationCoverage={citationCoverage}
                    onClaimClick={handleClaimClick}
                  />
                </div>
              ) : running ? (
                <div className="bg-surface rounded-xl border border-line p-8 flex items-center gap-4 text-sm text-muted font-body">
                  <span className="w-5 h-5 rounded-full border-2 border-brass border-t-transparent animate-spin flex-shrink-0" />
                  Generating investment memo…
                </div>
              ) : null}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
