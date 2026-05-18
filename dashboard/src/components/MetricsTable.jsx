import styles from './MetricsTable.module.css'

const COLS = [
  { key: 'model', label: 'Modelo' },
  { key: 'dataset_type', label: 'Tipo' },
  { key: 'accuracy', label: 'Accuracy' },
  { key: 'precision_macro', label: 'Precision Macro' },
  { key: 'recall_macro', label: 'Recall Macro' },
  { key: 'f1_macro', label: 'F1 Macro' },
  { key: 'kappa', label: 'Kappa' },
  { key: 'roc_auc_ovr_macro', label: 'ROC-AUC' },
]

function pct(val) {
  if (val === null || val === undefined) return '—'
  return `${(val * 100).toFixed(1)}%`
}

function colorScale(val) {
  if (val === null || val === undefined) return {}
  const hue = val * 120
  return { color: `hsl(${hue}, 70%, 65%)` }
}

export default function MetricsTable({ rows, bestModel, activeModel, onSelect }) {
  return (
    <div className={styles.wrapper}>
      <table className={styles.table}>
        <thead>
          <tr>
            {COLS.map(c => (
              <th key={c.key}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const key = `${row.model}__${row.dataset_type}`
            const isBest = bestModel?.model === row.model && bestModel?.dataset_type === row.dataset_type
            const isActive = activeModel === key
            return (
              <tr
                key={i}
                className={`${isBest ? styles.best : ''} ${isActive ? styles.active : ''}`}
                onClick={() => onSelect(key)}
              >
                <td>
                  <span className={styles.modelName}>{row.model?.replace(/_/g, ' ')}</span>
                  {isBest && <span className={styles.bestBadge}>★ Mejor</span>}
                </td>
                <td><span className={`${styles.typeBadge} ${styles[row.dataset_type]}`}>{row.dataset_type}</span></td>
                <td style={colorScale(row.accuracy)}>{pct(row.accuracy)}</td>
                <td style={colorScale(row.precision_macro)}>{pct(row.precision_macro)}</td>
                <td style={colorScale(row.recall_macro)}>{pct(row.recall_macro)}</td>
                <td style={colorScale(row.f1_macro)}>{pct(row.f1_macro)}</td>
                <td style={colorScale(row.kappa ? (row.kappa + 1) / 2 : null)}>{row.kappa !== null && row.kappa !== undefined ? row.kappa.toFixed(3) : '—'}</td>
                <td style={colorScale(row.roc_auc_ovr_macro)}>{pct(row.roc_auc_ovr_macro)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
