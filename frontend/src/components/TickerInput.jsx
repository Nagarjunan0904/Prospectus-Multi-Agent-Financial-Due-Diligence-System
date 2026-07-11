const EXAMPLE_TICKERS = ['NVDA', 'AAPL', 'MSFT', 'AMD', 'GOOGL', 'TSLA']

export default function TickerInput({ value, onChange, onSubmit, disabled }) {
  const handleKey = (e) => {
    if (e.key === 'Enter' && value.trim()) onSubmit()
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-3">
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value.toUpperCase())}
          onKeyDown={handleKey}
          placeholder="TICKER"
          disabled={disabled}
          style={{ caretColor: 'var(--color-brass)' }}
          className="flex-1 px-4 py-2.5 bg-ink border border-line rounded-lg
                     font-mono text-sm text-ink-text placeholder:text-line
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors duration-150 input-glow"
        />
        <button
          onClick={() => onSubmit()}
          disabled={disabled || !value.trim()}
          className="px-6 py-2.5 bg-brass text-ink font-display font-semibold text-sm
                     rounded-lg tracking-wide
                     hover:brightness-110 hover:scale-[1.02]
                     active:scale-[0.97] active:brightness-95
                     disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100
                     transition-all duration-150 shadow-sm"
        >
          {disabled ? 'Running…' : 'Analyse'}
        </button>
      </div>

      <div className="flex gap-2 flex-wrap">
        {EXAMPLE_TICKERS.map((t) => (
          <button
            key={t}
            onClick={() => { onChange(t); onSubmit(t) }}
            disabled={disabled}
            className="px-3 py-1 rounded-full text-xs font-mono font-medium
                       border border-line text-brass
                       hover:bg-brass/10 hover:border-brass/40
                       disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all duration-150"
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  )
}
