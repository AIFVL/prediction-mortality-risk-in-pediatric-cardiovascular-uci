import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell
} from 'recharts'
import styles from './ProbabilityChart.module.css'

const COLORS = ['#34d399', '#fb923c']
const DEFAULT_LABELS = { '0': 'No murió', '1': 'Murió' }

export default function ProbabilityChart({ model, classes }) {
  const labels = classes || DEFAULT_LABELS
  const proba = model.avg_prediction_probabilities || {}

  const data = Object.entries(proba)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([key, val], i) => ({
      name: labels[key.replace('class_', '')] || `Clase ${key.replace('class_', '')}`,
      value: parseFloat((val * 100).toFixed(1)),
      fill: COLORS[i] || '#94a3b8',
    }))

  if (data.length === 0) {
    return (
      <div className={styles.card}>
        <h3 className={styles.title}>Probabilidad Promedio por Clase</h3>
        <p className={styles.noData}>Sin datos de probabilidad disponibles.</p>
      </div>
    )
  }

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Probabilidad Promedio por Clase</h3>
      <p className={styles.subtitle}>
        Probabilidad media que el modelo asigna a cada clase en el conjunto de prueba
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: -10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            domain={[0, 100]}
            tickFormatter={v => `${v}%`}
          />
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
            formatter={(value) => [`${value}%`, 'Probabilidad media']}
            labelStyle={{ color: '#f1f5f9', fontWeight: 600 }}
          />
          <Bar dataKey="value" radius={[6, 6, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className={styles.legend}>
        {data.map((d, i) => (
          <div key={i} className={styles.legendItem}>
            <span className={styles.dot} style={{ background: d.fill }} />
            <span>{d.name}: <strong>{d.value}%</strong></span>
          </div>
        ))}
      </div>
    </div>
  )
}
