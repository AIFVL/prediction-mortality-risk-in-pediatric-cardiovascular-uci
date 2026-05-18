import styles from './PerClassMetrics.module.css'

const DEFAULT_LABELS = { '0': 'No murió', '1': 'Murió' }

function pct(val) {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return '—'
  return `${(Number(val) * 100).toFixed(1)}%`
}

function classNameFor(row, labels) {
  const key = String(Number(row.class))
  return `${key} — ${labels[key] || `Clase ${key}`}`
}

export default function PerClassMetrics({ rows, classes, title = 'Métricas por Clase' }) {
  if (!rows || rows.length === 0) return null
  const labels = classes || DEFAULT_LABELS

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>{title}</h3>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Clase</th>
              <th>Precisión</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Sensibilidad</th>
              <th>Especificidad</th>
              <th>ROC-AUC</th>
              <th>Soporte</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                <td><span className={styles.classBadge}>{classNameFor(row, labels)}</span></td>
                <td>{pct(row.precision)}</td>
                <td>{pct(row.recall)}</td>
                <td>{pct(row.f1_score)}</td>
                <td>{pct(row.sensitivity)}</td>
                <td>{pct(row.specificity)}</td>
                <td>{pct(row.roc_auc)}</td>
                <td className={styles.support}>{Number(row.support).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
