// severity → dot color using brand palette (high=rose, medium=brass, low=muted)
const SEV_DOT = {
  high:   'bg-rose',
  medium: 'bg-brass',
  low:    'bg-muted',
}

const SEV_BADGE = {
  high:   'text-rose   border-rose/30   bg-rose/10',
  medium: 'text-brass  border-brass/30  bg-brass/10',
  low:    'text-muted  border-line      bg-surface',
}

export default function RiskFlagList({ flags }) {
  if (!flags) return null

  return (
    <div className="rounded-xl border border-line bg-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-line flex items-center gap-2">
        <span className="text-xs font-mono text-muted uppercase tracking-widest">Risk Flags</span>
        {flags.length > 0 && (
          <span className="text-xs font-mono px-1.5 py-0.5 rounded-full bg-rose/10 text-rose border border-rose/30 font-semibold">
            {flags.length}
          </span>
        )}
      </div>

      {flags.length === 0 ? (
        <div className="px-4 py-6 text-sm text-muted italic font-body text-center">
          No risk flags detected.
        </div>
      ) : (
        <ul className="divide-y divide-line/60">
          {flags.map((flag, i) => {
            const sev = flag.severity?.toLowerCase() || 'low'
            return (
              <li key={i} className="px-4 py-3 space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${SEV_DOT[sev] || 'bg-muted'}`} />
                  <span className="font-mono text-xs font-medium text-ink-text flex-1 truncate">
                    {flag.flag}
                  </span>
                  <span
                    className={`text-[10px] font-mono px-2 py-0.5 rounded-full border font-semibold flex-shrink-0 ${
                      SEV_BADGE[sev] || SEV_BADGE.low
                    }`}
                  >
                    {sev.toUpperCase()}
                  </span>
                </div>
                {flag.evidence && (
                  <p className="font-body text-xs text-muted pl-5 leading-relaxed">
                    {flag.evidence}
                  </p>
                )}
                {flag.source_tool && (
                  <p className="font-mono text-[10px] text-line pl-5">
                    {flag.source_tool}
                  </p>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}
