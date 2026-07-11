const RATIO_META = [
  { key: 'pe_ratio',           label: 'P/E Ratio',           pct: false },
  { key: 'debt_to_equity',     label: 'Debt / Equity',       pct: false },
  { key: 'current_ratio',      label: 'Current Ratio',       pct: false },
  { key: 'gross_margin',       label: 'Gross Margin',        pct: true  },
  { key: 'operating_margin',   label: 'Operating Margin',    pct: true  },
  { key: 'net_margin',         label: 'Net Margin',          pct: true  },
  { key: 'revenue_growth_yoy', label: 'Revenue Growth YoY',  pct: true  },
  { key: 'revenue_growth_qoq', label: 'Revenue Growth QoQ',  pct: true  },
]

function fmt(val, pct) {
  if (val == null) return '—'
  return pct ? `${(val * 100).toFixed(1)}%` : val.toFixed(2)
}

export default function RatioTable({ ratios }) {
  if (!ratios) return null

  const rows = RATIO_META.filter(({ key }) => ratios[key] != null)

  return (
    <div className="rounded-xl border border-line bg-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-line">
        <span className="text-xs font-mono text-muted uppercase tracking-widest">Financial Ratios</span>
      </div>
      <table className="w-full text-sm">
        <tbody>
          {rows.map(({ key, label, pct }, i) => (
            <tr
              key={key}
              className={i % 2 === 0 ? 'bg-surface' : 'bg-ink/40'}
            >
              <td className="px-4 py-2.5 font-body text-muted text-xs">{label}</td>
              <td className="px-4 py-2.5 text-right font-mono font-medium text-ink-text text-sm tabular-nums">
                {fmt(ratios[key], pct)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {ratios.warnings?.length > 0 && (
        <div className="border-t-2 border-rose/40 bg-rose/5 px-4 py-2.5 space-y-0.5">
          {ratios.warnings.map((w, i) => (
            <div key={i} className="text-xs font-body text-rose/80">{w}</div>
          ))}
        </div>
      )}
    </div>
  )
}
