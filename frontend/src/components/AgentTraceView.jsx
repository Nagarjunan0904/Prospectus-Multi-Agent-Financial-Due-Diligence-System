import { forwardRef, useImperativeHandle, useRef, useMemo } from 'react'

function fmtLatency(ms) {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}

function latencyColor(ms) {
  if (ms < 500)  return 'text-muted'
  if (ms < 2000) return 'text-brass'
  return 'text-amber-400'
}

const STATUS_DOT = {
  success: 'bg-mint',
  error:   'bg-rose',
  skipped: 'bg-line',
}

const AgentTraceView = forwardRef(function AgentTraceView({ entries }, ref) {
  const groupRefs = useRef({})

  // Group consecutive entries by node — stable firstIndex keys prevent remounts of
  // existing groups, so CSS animations only fire for newly added groups.
  const groups = useMemo(() => {
    const result = []
    entries.forEach((entry, i) => {
      if (!result.length || result[result.length - 1].node !== entry.node) {
        result.push({ node: entry.node, firstIndex: i, items: [] })
      }
      result[result.length - 1].items.push({ ...entry, originalIndex: i })
    })
    return result
  }, [entries])

  useImperativeHandle(ref, () => ({
    scrollToNode(nodeName) {
      const match = Object.values(groupRefs.current).find(
        (el) => el?.dataset?.node === nodeName
      )
      if (match) {
        match.scrollIntoView({ behavior: 'smooth', block: 'center' })
        match.classList.add('highlight-node')
        setTimeout(() => match.classList.remove('highlight-node'), 1500)
      }
    },
  }))

  if (!entries.length) {
    return (
      <div className="rounded-xl border border-line bg-surface px-4 py-8 text-center text-sm text-muted font-body">
        Pipeline trace will appear here.
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-line bg-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-line">
        <span className="text-xs font-mono text-muted uppercase tracking-widest">
          Pipeline Trace
        </span>
      </div>

      <div className="p-4 max-h-[420px] overflow-y-auto">
        {groups.map((group, gi) => {
          const isLast    = gi === groups.length - 1
          const hasError  = group.items.some((e) => e.status === 'error')
          const hasSuccess = group.items.some((e) => e.status === 'success')
          const dotColor  = hasError ? 'bg-rose' : hasSuccess ? 'bg-mint' : 'bg-brass'

          return (
            <div
              key={group.firstIndex}
              ref={(el) => { groupRefs.current[group.firstIndex] = el }}
              data-node={group.node}
              className="relative pl-8 pb-3"
            >
              {/* Connector line to next node — animates in when this group is no longer last */}
              {!isLast && (
                <div className="absolute left-[7px] top-4 bottom-0 w-0.5 bg-line animate-line-draw" />
              )}

              {/* Node dot — pulses brass ring on mount, color reflects current status */}
              <div
                className={`absolute left-0 top-0.5 w-3.5 h-3.5 rounded-full
                             animate-dot-pulse transition-colors duration-500 ${dotColor}`}
              />

              {/* Node label */}
              <div className="text-xs font-mono font-semibold text-brass leading-none mb-2">
                {group.node}
              </div>

              {/* Tool-call sub-entries */}
              <div className="space-y-1.5">
                {group.items.map((entry) => (
                  <div key={entry.originalIndex} className="flex items-center gap-2 pl-1">
                    <div className="w-3 h-px bg-line flex-shrink-0" />
                    <span
                      className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                        STATUS_DOT[entry.status] || 'bg-line'
                      }`}
                    />
                    <span className="text-xs font-mono text-ink-text/70 flex-1 truncate">
                      {entry.tool || '—'}
                    </span>
                    {entry.latency_ms != null && (
                      <span className={`text-xs font-mono flex-shrink-0 ${latencyColor(entry.latency_ms)}`}>
                        {fmtLatency(entry.latency_ms)}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
})

export default AgentTraceView
