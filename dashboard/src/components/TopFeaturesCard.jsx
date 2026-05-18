import styles from './TopFeaturesCard.module.css'

export default function TopFeaturesCard({ featureImportance }) {
  const features = featureImportance?.features ?? []

  if (features.length === 0) {
    return (
      <div className={styles.card}>
        <h3 className={styles.title}>Variables más importantes</h3>
        <p className={styles.noData}>
          Sin datos de importancia para este modelo. Ejecuta el pipeline completo.
        </p>
      </div>
    )
  }

  const maxImp = Math.max(...features.map(f => f.importance), 0.0001)
  const methodLabel = featureImportance?.method_label || featureImportance?.method

  return (
    <div className={styles.card}>
      <h3 className={styles.title}>Variables más importantes</h3>
      <p className={styles.subtitle}>
        Top {features.length} según el modelo entrenado
        {methodLabel ? ` · ${methodLabel}` : ''}
      </p>
      <ul className={styles.list}>
        {features.map(item => (
          <li key={item.rank} className={styles.row}>
            <span className={styles.rank}>{item.rank}</span>
            <div className={styles.body}>
              <span className={styles.name} title={item.variable}>
                {item.variable}
              </span>
              <div className={styles.barTrack}>
                <div
                  className={styles.barFill}
                  style={{ width: `${(item.importance / maxImp) * 100}%` }}
                />
              </div>
            </div>
            <span className={styles.value}>{(item.importance * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
