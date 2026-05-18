import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import styles from './MetricsChart.module.css'

const METRICS = [
  { key: 'accuracy', label: 'Accuracy', color: '#38bdf8' },
  { key: 'recall_macro', label: 'Recall Macro', color: '#a78bfa' },
  { key: 'precision_macro', label: 'Precision Macro', color: '#34d399' },
  { key: 'f1_macro', label: 'F1 Macro', color: '#818cf8' },
  { key: 'kappa', label: 'Kappa', color: '#fb923c' },
]

function formatModelName(model, type) {
  const names = {
    logistic_regression: 'LR',
    random_forest: 'RF',
    svm: 'SVM',
    xgboost: 'XGB',
  }
  return `${names[model] || model} [${type}]`
}

export default function MetricsChart({ models }) {
  const chartData = models.map(m => {
    const entry = { name: formatModelName(m.model, m.dataset_type) }
    METRICS.forEach(metric => {
      const val = m.metrics[metric.key]
      entry[metric.key] = val !== null && val !== undefined ? parseFloat((val * 100).toFixed(1)) : null
    })
    return entry
  })

  return (
    <div className={styles.card}>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            domain={[0, 100]}
            tickFormatter={v => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={(value) => [`${value}%`]}
            labelStyle={{ color: '#f1f5f9', fontWeight: 600 }}
          />
          <Legend
            wrapperStyle={{ fontSize: 12, paddingTop: 12 }}
            formatter={(value) => METRICS.find(m => m.key === value)?.label || value}
          />
          {METRICS.map(m => (
            <Bar key={m.key} dataKey={m.key} name={m.key} fill={m.color} radius={[4, 4, 0, 0]} maxBarSize={30} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
