import styles from './ConfusionMatrix.module.css'

const DEFAULT_LABELS = { '0': 'No murió', '1': 'Murió' }

export default function ConfusionMatrix({ matrix, classes }) {
  const labels = classes || DEFAULT_LABELS
  const normalizedMatrix = matrix.map(row => row.map(val => Number(val) || 0))
  const n = Math.max(
    normalizedMatrix.length,
    ...normalizedMatrix.map(row => row.length),
  )
  const classLabels = Object.values(labels).slice(0, n)

  const completeMatrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => normalizedMatrix[i]?.[j] ?? 0),
  )

  const maxVal = Math.max(...completeMatrix.flat(), 0)

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Matriz de Confusión</h3>
      <p className={styles.subtitle}>Filas = Real · Columnas = Predicho</p>
      <div className={styles.tableWrap}>
        <table className={styles.matrix}>
          <thead>
            <tr>
              <th className={styles.corner}></th>
              {classLabels.map((l, i) => (
                <th key={i} className={styles.colHeader}>{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {completeMatrix.map((row, i) => (
              <tr key={i}>
                <th className={styles.rowHeader}>{classLabels[i] || i}</th>
                {row.map((val, j) => {
                  const isDiag = i === j
                  const intensity = maxVal > 0 ? val / maxVal : 0
                  const bg = isDiag
                    ? `rgba(52, 211, 153, ${0.15 + intensity * 0.65})`
                    : val > 0 ? `rgba(248, 113, 113, ${0.1 + intensity * 0.5})` : 'transparent'
                  return (
                    <td
                      key={j}
                      className={styles.cell}
                      style={{ background: bg }}
                    >
                      <span className={`${styles.val} ${isDiag ? styles.diag : ''}`}>{val}</span>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
