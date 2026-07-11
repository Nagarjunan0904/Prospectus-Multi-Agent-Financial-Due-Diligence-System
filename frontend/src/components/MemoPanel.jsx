const SOURCE_FIELD_TO_AGENT = {
  ratios:          'quant_agent',
  ratio_history:   'quant_agent',
  sentiment:       'sentiment_agent',
  risk_flags:      'risk_agent',
  company_facts:   'data_agent',
  filing_sections: 'data_agent',
  insider_summary: 'data_agent',
}

function deriveAgent(sourceField) {
  if (!sourceField) return null
  return SOURCE_FIELD_TO_AGENT[sourceField.split('.')[0]] || null
}

// Per-section accent colors (border + heading)
const SECTION_META = {
  'Financial Snapshot': {
    border:   'border-brass',
    heading:  'text-brass',
    chipHover: 'hover:border-brass/60',
  },
  'Sentiment': {
    border:   'border-mint',
    heading:  'text-mint',
    chipHover: 'hover:border-mint/60',
  },
  'Risk Factors': {
    border:   'border-rose',
    heading:  'text-rose',
    chipHover: 'hover:border-rose/60',
  },
  'Recommendation': {
    border:   'border-brass',
    heading:  'text-brass',
    chipHover: 'hover:border-brass/60',
  },
}

const DEFAULT_META = { border: 'border-line', heading: 'text-ink-text', chipHover: 'hover:border-line' }

export default function MemoPanel({ memo, citationCoverage, onClaimClick }) {
  if (!memo) return null

  const { ticker, sections = [] } = memo

  return (
    <div className="space-y-5">
      {/* Header row */}
      <div className="flex items-center gap-3 pb-2 border-b border-line">
        <h2 className="font-display text-2xl font-bold text-ink-text">{ticker}</h2>
        {citationCoverage != null && (
          <span
            className={`text-xs font-mono px-2.5 py-1 rounded-full border font-semibold ${
              citationCoverage >= 0.8
                ? 'bg-mint/10 text-mint border-mint/30'
                : citationCoverage >= 0.5
                ? 'bg-brass/10 text-brass border-brass/30'
                : 'bg-rose/10 text-rose border-rose/30'
            }`}
          >
            {(citationCoverage * 100).toFixed(0)}% coverage
          </span>
        )}
      </div>

      {/* Sections */}
      {sections.map((section) => {
        const meta = SECTION_META[section.heading] || DEFAULT_META
        return (
          <div
            key={section.heading}
            className={`rounded-lg border-l-2 ${meta.border} bg-ink/40 border border-line/60 border-l-[3px] p-4 space-y-3`}
          >
            <h3 className={`font-display text-sm font-semibold ${meta.heading}`}>
              {section.heading}
            </h3>

            {section.claims?.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {section.claims.map((claim, i) => {
                  const agent = deriveAgent(claim.source_field)
                  return (
                    <button
                      key={i}
                      onClick={() => agent && onClaimClick && onClaimClick(agent)}
                      className={`group text-left px-3 py-2 rounded-lg border border-line bg-surface
                                  ${meta.chipHover} transition-colors duration-150
                                  ${agent ? 'cursor-pointer' : 'cursor-default'}`}
                      title={claim.source_field || undefined}
                    >
                      <div className="font-body text-xs text-ink-text leading-snug">
                        {claim.text}
                      </div>
                      {claim.source_field && (
                        <div className="font-mono text-[10px] text-muted mt-1 leading-none">
                          {claim.source_field}
                        </div>
                      )}
                    </button>
                  )
                })}
              </div>
            ) : (
              <p className="text-xs text-muted italic font-body">No claims in this section.</p>
            )}
          </div>
        )
      })}
    </div>
  )
}
