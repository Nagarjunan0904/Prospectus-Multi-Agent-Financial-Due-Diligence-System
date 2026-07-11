const AGENTS = [
  {
    name: 'DATA AGENT',
    desc: 'SEC EDGAR filings, XBRL financials, insider trades',
  },
  {
    name: 'QUANT AGENT',
    desc: 'Ratio computation, peer comparison, historical trends',
  },
  {
    name: 'SENTIMENT AGENT',
    desc: 'FinBERT-scored news sentiment',
  },
  {
    name: 'RISK AGENT',
    desc: 'Debt spikes, insider clusters, audit-language flags',
  },
]

export default function AgentRoster() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {AGENTS.map(({ name, desc }) => (
        <div
          key={name}
          className="bg-surface rounded-xl border border-line px-4 py-3"
        >
          <div className="font-display text-[11px] font-semibold text-brass tracking-widest mb-1.5 uppercase">
            {name}
          </div>
          <div className="font-body text-xs text-muted leading-relaxed">
            {desc}
          </div>
        </div>
      ))}
    </div>
  )
}
