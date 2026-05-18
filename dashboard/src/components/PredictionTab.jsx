import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import styles from './PredictionTab.module.css'

const MODEL_LABELS = {
  logistic_regression: 'Logistic Regression',
  random_forest: 'Random Forest',
  svm: 'SVM',
  xgboost: 'XGBoost',
}

function UploadZone({ onUpload, loading }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file) return
    onUpload(file)
  }, [onUpload])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  return (
    <div
      className={`${styles.dropzone} ${dragging ? styles.dragging : ''} ${loading ? styles.loading : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !loading && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls,.txt"
        style={{ display: 'none' }}
        onChange={(e) => handleFile(e.target.files[0])}
      />
      {loading ? (
        <>
          <div className={styles.dropSpinner} />
          <p className={styles.dropTitle}>Procesando pacientes...</p>
        </>
      ) : (
        <>
          <div className={styles.dropIcon}>📂</div>
          <p className={styles.dropTitle}>Arrastra tu archivo aquí o haz clic para seleccionar</p>
          <p className={styles.dropSub}>Formatos soportados: CSV · XLSX · TXT</p>
          <p className={styles.dropHint}>
            El archivo debe contener las mismas variables del dataset de entrenamiento.
          </p>
        </>
      )}
    </div>
  )
}

function SummaryCard({ label, value, sub, accent }) {
  return (
    <div className={styles.summaryCard} style={accent ? { borderColor: accent } : {}}>
      <span className={styles.summaryValue} style={accent ? { color: accent } : {}}>{value}</span>
      <span className={styles.summaryLabel}>{label}</span>
      {sub && <span className={styles.summarySub}>{sub}</span>}
    </div>
  )
}

function ModelCard({ mp, totalPatients }) {
  const pct = totalPatients > 0 ? Math.round((mp.n_predicted_died / totalPatients) * 100) : 0
  const survPct = 100 - pct

  return (
    <div className={`${styles.modelCard} ${mp.is_best ? styles.bestCard : ''}`}>
      {mp.is_best && <span className={styles.bestBadge}>Mejor Modelo</span>}
      <div className={styles.modelCardHeader}>
        <span className={styles.modelCardName}>{MODEL_LABELS[mp.model] || mp.model}</span>        <span className={`${styles.modelCardType} ${mp.dataset_type === 'balanced' ? styles.balancedTag : styles.rawTag}`}>
          {mp.dataset_type.toUpperCase()}
        </span>
      </div>

      {mp.error ? (
        <p className={styles.modelError}>Error: {mp.error}</p>
      ) : (
        <>
          <div className={styles.modelBarWrap}>
            <div className={styles.modelBar}>
              <div className={styles.barDied} style={{ width: `${pct}%` }} />
              <div className={styles.barSurv} style={{ width: `${survPct}%` }} />
            </div>
            <div className={styles.barLegend}>
              <span className={styles.diedDot} />
              <span>{mp.n_predicted_died} murió ({pct}%)</span>
              <span className={styles.survDot} />
              <span>{mp.n_predicted_survived} sobrevivió ({survPct}%)</span>
            </div>
          </div>
          {mp.avg_prob_died != null && (
            <p className={styles.avgProb}>
              Prob. media de morir: <strong>{(mp.avg_prob_died * 100).toFixed(1)}%</strong>
            </p>
          )}
        </>
      )}
    </div>
  )
}

function PatientTable({ modelPrediction, patientRows, displayColumns }) {
  const [page, setPage] = useState(0)
  const PAGE_SIZE = 15
  const totalPages = Math.ceil(patientRows.length / PAGE_SIZE)
  const pageRows = patientRows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  if (!modelPrediction || modelPrediction.error) return null

  const preds = modelPrediction.predictions || []
  const probs = modelPrediction.probabilities || []

  return (
    <div className={styles.tableWrap}>
      <div className={styles.tableScroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>#</th>
              {displayColumns.map(col => (
                <th key={col}>{col}</th>
              ))}
              <th>Predicción</th>
              <th>Prob. Muerte</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => {
              const globalIdx = page * PAGE_SIZE + i
              const pred = preds[globalIdx]
              const prob = probs[globalIdx]
              return (
                <tr key={row.patient_idx}>
                  <td className={styles.idxCell}>{row.patient_idx}</td>
                  {displayColumns.map(col => (
                    <td key={col} className={styles.dataCell}>{row[col]}</td>
                  ))}
                  <td>
                    <span className={pred === 1 ? styles.predDied : styles.predSurv}>
                      {pred === 1 ? 'Murió' : 'Sobrevivió'}
                    </span>
                  </td>
                  <td className={styles.probCell}>
                    {prob != null ? (
                      <div className={styles.probBar}>
                        <div className={styles.probFill} style={{ width: `${Math.round(prob * 100)}%` }} />
                        <span className={styles.probLabel}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ) : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className={styles.pagination}>
          <button disabled={page === 0} onClick={() => setPage(p => p - 1)}>← Anterior</button>
          <span>Página {page + 1} de {totalPages}</span>
          <button disabled={page === totalPages - 1} onClick={() => setPage(p => p + 1)}>Siguiente →</button>
        </div>
      )}
    </div>
  )
}

export default function PredictionTab() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)
  const [activeModel, setActiveModel] = useState(null)

  const handleUpload = async (file) => {
    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post('/api/predict-upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      const data = res.data
      setResults(data)

      const bestPred = data.model_predictions?.find(m => m.is_best && !m.error)
      const firstValid = data.model_predictions?.find(m => !m.error)
      const initial = bestPred || firstValid
      if (initial) setActiveModel(`${initial.model}__${initial.dataset_type}`)
    } catch (err) {
      const msg = err.response?.data?.detail || 'Error al procesar el archivo.'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setError(null)
    setActiveModel(null)
  }

  const validModels = results?.model_predictions?.filter(m => !m.error) || []
  const selectedModel = validModels.find(m => `${m.model}__${m.dataset_type}` === activeModel)
  const bestPred = validModels.find(m => m.is_best)
  const totalDied = bestPred?.n_predicted_died ?? validModels[0]?.n_predicted_died ?? 0
  const totalSurv = bestPred?.n_predicted_survived ?? validModels[0]?.n_predicted_survived ?? 0

  return (
    <div className={styles.tab}>
      <div className={styles.tabHeader}>
        <div>
          <h2 className={styles.tabTitle}>Predicción de Nuevos Pacientes</h2>
          <p className={styles.tabSub}>
            Sube un archivo con datos clínicos de nuevos pacientes para obtener la probabilidad
            de mortalidad estimada por cada modelo entrenado.
          </p>
        </div>
        {results && (
          <button className={styles.resetBtn} onClick={handleReset}>
            Cargar otro archivo
          </button>
        )}
      </div>

      {!results && (
        <UploadZone onUpload={handleUpload} loading={loading} />
      )}

      {error && (
        <div className={styles.errorBox}>
          <strong>Error:</strong> {error}
          <button className={styles.errorClose} onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {results && (
        <>
          <div className={styles.summaryRow}>
            <SummaryCard
              label="Pacientes cargados"
              value={results.n_patients}
              sub={results.filename}
              accent="var(--accent)"
            />
            <SummaryCard
              label="Predichos: Murió"
              value={totalDied}
              sub={`${results.n_patients > 0 ? Math.round(totalDied / results.n_patients * 100) : 0}% del total`}
              accent="var(--danger)"
            />
            <SummaryCard
              label="Predichos: Sobrevivió"
              value={totalSurv}
              sub={`${results.n_patients > 0 ? Math.round(totalSurv / results.n_patients * 100) : 0}% del total`}
              accent="var(--accent3)"
            />
            <SummaryCard
              label="Modelos aplicados"
              value={validModels.length}
              sub={`de ${results.model_predictions?.length || 0} disponibles`}
            />
          </div>

          <section className={styles.section}>
            <h3 className={styles.sectionTitle}>Resumen por Modelo</h3>
            <div className={styles.modelGrid}>
              {results.model_predictions.map(mp => (
                <ModelCard
                  key={`${mp.model}__${mp.dataset_type}`}
                  mp={mp}
                  totalPatients={results.n_patients}
                />
              ))}
            </div>
          </section>

          <section className={styles.section}>
            <div className={styles.tableHeader}>
              <h3 className={styles.sectionTitle}>Predicciones por Paciente</h3>
              <div className={styles.modelSelector}>
                {validModels.map(m => {
                  const key = `${m.model}__${m.dataset_type}`
                  return (
                    <button
                      key={key}
                      className={`${styles.modelBtn} ${activeModel === key ? styles.modelActive : ''}`}
                      onClick={() => setActiveModel(key)}
                    >
                      {MODEL_LABELS[m.model] || m.model}
                      <span className={`${styles.typeTag} ${m.dataset_type === 'smote' ? styles.smoteTag : styles.rawTag}`}>
                        {m.dataset_type.toUpperCase()}
                      </span>
                      {m.is_best && <span className={styles.bestMini}>★</span>}
                    </button>
                  )
                })}
              </div>
            </div>

            <PatientTable
              modelPrediction={selectedModel}
              patientRows={results.patient_rows}
              displayColumns={results.display_columns}
            />
          </section>
        </>
      )}
    </div>
  )
}
