import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import styles from './ClassDistribution.module.css'

const COLORS = ['#38bdf8', '#fb923c', '#f87171']
const DEFAULT_LABELS = { '0': 'No murió', '1': 'Murió' }

export default function ClassDistribution({ distribution, classes }) {
  const labels = classes || DEFAULT_LABELS

  const makeData = (counts) =>
    Object.entries(counts || {}).map(([key, val], i) => ({
      name: labels[key] || `Clase ${key}`,
      value: val,
      fill: COLORS[i] || '#94a3b8',
    }))

  const rawData = makeData(distribution.raw)
  const balancedData = makeData(distribution.balanced)

  const PieSection = ({ data, title }) => (
    <div className={styles.pieSection}>
      <h4 className={styles.pieTitle}>{title}</h4>
      <ResponsiveContainer width="100%" height={180}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={45}
            outerRadius={75}
            paddingAngle={3}
            dataKey="value"
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.fill} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
            formatter={(value, name) => [value, name]}
          />
        </PieChart>
      </ResponsiveContainer>
      <div className={styles.legend}>
        {data.map((d, i) => {
          const total = data.reduce((s, x) => s + x.value, 0)
          const pct = total > 0 ? ((d.value / total) * 100).toFixed(1) : '0'
          return (
            <div key={i} className={styles.legendItem}>
              <span className={styles.dot} style={{ background: d.fill }} />
              <span className={styles.legendText}>
                {d.name}: <strong>{d.value}</strong> ({pct}%)
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )

  return (
    <div className={styles.section}>
      <h2 className={styles.sectionTitle}>Distribución de Clases</h2>
      <div className={styles.grid}>
        {rawData.length > 0 && <PieSection data={rawData} title="Datos originales (RAW)" />}
        {balancedData.length > 0 && <PieSection data={balancedData} title="Tras balanceo (ADASYN)" />}
      </div>
    </div>
  )
}
