import { useState, useEffect } from 'react'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Resolvers match the real /health response shape:
// { status, db, mcp_servers: { data, quant, sentiment, risk } }
const SERVICES = [
  { label: 'DB',        resolve: (h) => h.db },
  { label: 'Data',      resolve: (h) => h.mcp_servers?.data },
  { label: 'Quant',     resolve: (h) => h.mcp_servers?.quant },
  { label: 'Sentiment', resolve: (h) => h.mcp_servers?.sentiment },
  { label: 'Risk',      resolve: (h) => h.mcp_servers?.risk },
]

// Returns 'loading' | 'up' | 'down'
function dotStatus(health, networkError, resolve) {
  if (networkError)    return 'down'
  if (health === null) return 'loading'
  return resolve(health) === 'up' ? 'up' : 'down'
}

export default function SystemStatus() {
  const [health, setHealth]           = useState(null)
  const [networkError, setNetworkErr] = useState(false)

  useEffect(() => {
    let mounted = true

    async function poll() {
      try {
        const res = await fetch(`${API}/health`, {
          signal: AbortSignal.timeout(5000),
        })
        if (!mounted) return
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data = await res.json()
        if (!mounted) return
        setHealth(data)
        setNetworkErr(false)
      } catch {
        if (!mounted) return
        setNetworkErr(true)
      }
    }

    poll()
    const id = setInterval(poll, 30_000)
    return () => { mounted = false; clearInterval(id) }
  }, [])

  return (
    <div className="flex items-center gap-1 px-1">
      <span className="font-mono text-[10px] text-ink-text/60 uppercase tracking-widest mr-2 select-none">
        Services
      </span>
      {SERVICES.map(({ label, resolve }) => {
        const status = dotStatus(health, networkError, resolve)
        return (
          <div key={label} className="flex items-center gap-1.5 px-2 py-1">
            {status === 'loading' ? (
              <span className="w-1.5 h-1.5 rounded-full bg-muted/50 animate-pulse" />
            ) : status === 'up' ? (
              <span className="w-1.5 h-1.5 rounded-full bg-mint" />
            ) : (
              <span className="w-1.5 h-1.5 rounded-full bg-rose" />
            )}
            <span className="font-mono text-[11px] text-muted">{label}</span>
          </div>
        )
      })}
    </div>
  )
}
