import { RadialBarChart, RadialBar, Legend, ResponsiveContainer, Tooltip } from 'recharts'

export default function SentimentGauge({ sentiment }) {
  if (!sentiment) return null

  const {
    positive_pct  = 0,
    neutral_pct   = 0,
    negative_pct  = 0,
    headline_count = 0,
  } = sentiment

  const data = [
    { name: 'Positive', value: +(positive_pct * 100).toFixed(1), fill: '#3FB984' }, // mint
    { name: 'Neutral',  value: +(neutral_pct  * 100).toFixed(1), fill: '#8E98B0' }, // muted
    { name: 'Negative', value: +(negative_pct * 100).toFixed(1), fill: '#E2596B' }, // rose
  ]

  const dominant = data.reduce((a, b) => (a.value >= b.value ? a : b))

  return (
    <div className="rounded-xl border border-line bg-surface overflow-hidden">
      <div className="px-4 py-3 border-b border-line">
        <span className="text-xs font-mono text-muted uppercase tracking-widest">
          News Sentiment
        </span>
        {headline_count > 0 && (
          <span className="ml-2 font-mono text-xs text-muted">
            · {headline_count} headlines
          </span>
        )}
      </div>

      <div className="px-4 pb-4">
        <div className="h-44">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              cx="50%"
              cy="55%"
              innerRadius="30%"
              outerRadius="82%"
              startAngle={180}
              endAngle={0}
              data={data}
              barSize={13}
            >
              <RadialBar
                dataKey="value"
                cornerRadius={5}
                background={{ fill: '#26324A' }}
              />
              <Tooltip
                contentStyle={{
                  background: '#141B2D',
                  border: '1px solid #26324A',
                  borderRadius: 8,
                  fontSize: 12,
                  fontFamily: "'IBM Plex Mono', monospace",
                  color: '#E8ECF6',
                }}
                formatter={(v) => [`${v}%`]}
                cursor={false}
              />
              <Legend
                iconSize={8}
                iconType="circle"
                wrapperStyle={{
                  fontSize: 11,
                  fontFamily: "'IBM Plex Mono', monospace",
                  color: '#8E98B0',
                  paddingTop: 6,
                }}
              />
            </RadialBarChart>
          </ResponsiveContainer>
        </div>

        {/* Dominant sentiment label */}
        <div className="text-center -mt-1">
          <span className="font-display text-sm font-semibold" style={{ color: dominant.fill }}>
            {dominant.name}
          </span>
          <span className="font-mono text-xs text-muted ml-1.5">
            {dominant.value}%
          </span>
        </div>
      </div>
    </div>
  )
}
