import styles from './BestModelCard.module.css'

function MetricItem({ label, value, isDecimal = false }) {
  let displayValue = 'N/A'
  if (value !== null && value !== undefined) {
    displayValue = isDecimal ? value.toFixed(3) : `${(value * 100).toFixed(1)}%`
  }
  return (
    <div className={styles.metric}>
      <span className={styles.metricValue}>{displayValue}</span>
      <span className={styles.metricLabel}>{label}</span>
    </div>
  )
}

export default function BestModelCard({ best, classes }) {
  return (
    <div className={styles.card}>
      <div className={styles.crown}>🏆</div>
      <div className={styles.content}>
        <div className={styles.top}>
          <div>
            <div className={styles.tag}>Mejor Modelo</div>
            <h2 className={styles.name}>{best.model?.replace(/_/g, ' ')}</h2>
            <span className={styles.type}>{best.dataset_type?.toUpperCase()}</span>
          </div>
          <div className={styles.metrics}>
            <MetricItem label="Recall Macro" value={best.recall_macro} />
            <MetricItem label="Accuracy" value={best.accuracy} />
            <MetricItem label="Precision Macro" value={best.precision_macro} />
            <MetricItem label="F1 Macro" value={best.f1_macro} />
            <MetricItem label="Kappa" value={best.kappa} isDecimal={true} />
            {best.roc_auc_ovr_macro !== null && (
              <MetricItem label="ROC-AUC" value={best.roc_auc_ovr_macro} />
            )}
          </div>
        </div>
        <p className={styles.description}>
          Este modelo obtuvo el mayor <strong>Recall Macro</strong> en el conjunto de prueba,
          priorizando la sensibilidad para detectar correctamente los casos críticos.
        </p>
      </div>
    </div>
  )
}
