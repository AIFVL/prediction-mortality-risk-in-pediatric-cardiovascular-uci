import styles from './Rachs1Comparison.module.css'

function pct(val) {
  if (val === null || val === undefined) return '—'
  return `${(val * 100).toFixed(1)}%`
}

function fmt3(val) {
  if (val === null || val === undefined) return '—'
  return val.toFixed(3)
}

function colorScale(val) {
  if (val === null || val === undefined) return {}
  const hue = val * 120
  return { color: `hsl(${hue}, 70%, 65%)` }
}

const METRIC_ROWS = [
  { key: 'accuracy',           label: 'Accuracy',            fmt: pct },
  { key: 'precision_macro',    label: 'Precision Macro',     fmt: pct },
  { key: 'recall_macro',       label: 'Recall Macro',        fmt: pct },
  { key: 'f1_macro',           label: 'F1 Macro',            fmt: pct },
  { key: 'kappa',              label: "Cohen's Kappa",       fmt: fmt3 },
  { key: 'roc_auc',            label: 'ROC-AUC',             fmt: pct },
]

function ConfusionMini({ matrix, title, subtitle }) {
  if (!matrix || matrix.length === 0) return null
  const labels = ['No murió', 'Murió']
  const maxVal = Math.max(...matrix.flat())
  return (
    <div className={styles.confusionWrap}>
      <div className={styles.confusionTitle}>{title}</div>
      {subtitle && <div className={styles.confusionSubtitle}>{subtitle}</div>}
      <table className={styles.cm}>
        <thead>
          <tr>
            <th></th>
            {labels.map((l, i) => <th key={i} className={styles.cmHead}>{l}</th>)}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className={styles.cmRowHead}>{labels[i]}</th>
              {row.map((val, j) => {
                const isDiag = i === j
                const intensity = maxVal > 0 ? val / maxVal : 0
                const bg = isDiag
                  ? `rgba(52, 211, 153, ${0.15 + intensity * 0.65})`
                  : val > 0 ? `rgba(248, 113, 113, ${0.1 + intensity * 0.5})` : 'transparent'
                return (
                  <td key={j} className={styles.cmCell} style={{ background: bg }}>
                    <span className={`${styles.cmVal} ${isDiag ? styles.diag : ''}`}>{val}</span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Rachs1Comparison({ rachs1, bestModel, models }) {
  if (!rachs1) return null

  const { metrics, confusion_matrix: cm, per_class_metrics, class_distribution, n_valid_rachs1, n_excluded_no_aplica } = rachs1

  const bestEntry = models?.find(
    m => m.model === bestModel?.model && m.dataset_type === bestModel?.dataset_type
  )
  const bestMetrics = bestEntry?.metrics || null
  const bestCm = bestEntry?.confusion_matrix || null

  const rachs1RocKey = 'roc_auc'
  const mlRocKey = 'roc_auc_ovr_macro'

  return (
    <section className={styles.section}>
      <div className={styles.header}>
        <div>
          <h2 className={styles.title}>Comparación con RACHS-1</h2>
          <p className={styles.subtitle}>
            Escala de riesgo clínico RACHS-1 vs mejor modelo de ML.
            {' '}<span className={styles.badge}>{n_valid_rachs1} pacientes con RACHS-1</span>
            {n_excluded_no_aplica > 0 && (
              <span className={styles.badgeMuted}> · {n_excluded_no_aplica} excluidos (NO APLICA)</span>
            )}
          </p>
        </div>
        <div className={styles.rachs1Info}>
          <span className={styles.ruleTag}>Regla RACHS-1</span>
          <span className={styles.ruleText}>Score ≤ 3 → No murió · Score &gt; 3 → Murió</span>
        </div>
      </div>

      <div className={styles.tableCard}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.metricCol}>Métrica</th>
              <th className={styles.valueCol}>
                <span className={styles.rachs1Label}>RACHS-1</span>
              </th>
              {bestMetrics && (
                <th className={styles.valueCol}>
                  <span className={styles.mlLabel}>
                    {bestModel?.model?.replace(/_/g, ' ')} [{bestModel?.dataset_type}]
                  </span>
                </th>
              )}
              {bestMetrics && <th className={styles.deltaCol}>Δ (ML − RACHS-1)</th>}
            </tr>
          </thead>
          <tbody>
            {METRIC_ROWS.map(({ key, label, fmt }) => {
              const r1Val = key === rachs1RocKey ? metrics?.[key] : metrics?.[key]
              const mlVal = key === rachs1RocKey
                ? bestMetrics?.[mlRocKey]
                : bestMetrics?.[key]

              const r1Num = typeof r1Val === 'number' ? r1Val : null
              const mlNum = typeof mlVal === 'number' ? mlVal : null
              const delta = r1Num !== null && mlNum !== null ? mlNum - r1Num : null

              const scaleVal = key === 'mcc' || key === 'kappa'
                ? (r1Num !== null ? (r1Num + 1) / 2 : null)
                : r1Num
              const scaleValMl = key === 'mcc' || key === 'kappa'
                ? (mlNum !== null ? (mlNum + 1) / 2 : null)
                : mlNum

              return (
                <tr key={key}>
                  <td className={styles.metricLabel}>{label}</td>
                  <td className={styles.valueCell} style={colorScale(scaleVal)}>
                    {fmt(r1Num)}
                  </td>
                  {bestMetrics && (
                    <td className={styles.valueCell} style={colorScale(scaleValMl)}>
                      {fmt(mlNum)}
                    </td>
                  )}
                  {bestMetrics && (
                    <td className={`${styles.deltaCell} ${delta === null ? '' : delta > 0 ? styles.pos : delta < 0 ? styles.neg : styles.zero}`}>
                      {delta === null ? '—' : `${delta > 0 ? '+' : ''}${key === 'mcc' || key === 'kappa' ? delta.toFixed(3) : pct(delta)}`}
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className={styles.cmRow}>
        <ConfusionMini
          matrix={cm}
          title="Matriz de Confusión — RACHS-1"
          subtitle="Filas = Real · Columnas = Predicho por RACHS-1"
        />
        {bestCm && bestModel && (
          <ConfusionMini
            matrix={bestCm}
            title={`Matriz de Confusión — ${bestModel.model?.replace(/_/g, ' ')} [${bestModel.dataset_type}]`}
            subtitle="Filas = Real · Columnas = Predicho por ML (Out-of-Fold)"
          />
        )}
      </div>

      {per_class_metrics && per_class_metrics.length > 0 && (
        <div className={styles.perClassCard}>
          <div className={styles.perClassTitle}>Métricas por Clase — RACHS-1</div>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Clase</th>
                <th>Precisión</th>
                <th>Recall</th>
                <th>F1</th>
                <th>Sensibilidad</th>
                <th>Especificidad</th>
                <th>Soporte</th>
              </tr>
            </thead>
            <tbody>
              {per_class_metrics.map((row, i) => (
                <tr key={i}>
                  <td><span className={styles.classBadge}>{row.class === 0 ? '0 — No murió' : '1 — Murió'}</span></td>
                  <td style={colorScale(row.precision)}>{pct(row.precision)}</td>
                  <td style={colorScale(row.recall)}>{pct(row.recall)}</td>
                  <td style={colorScale(row.f1_score)}>{pct(row.f1_score)}</td>
                  <td style={colorScale(row.sensitivity)}>{pct(row.sensitivity)}</td>
                  <td style={colorScale(row.specificity)}>{pct(row.specificity)}</td>
                  <td className={styles.support}>{row.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
