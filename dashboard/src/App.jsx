import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'

// Configuración de la URL base para el API en producción
const API_BASE_URL = import.meta.env.VITE_API_URL || '';
axios.defaults.baseURL = API_BASE_URL;

import Header from './components/Header.jsx'
import PipelineControl from './components/PipelineControl.jsx'
import BestModelCard from './components/BestModelCard.jsx'
import TopFeaturesCard from './components/TopFeaturesCard.jsx'
import MetricsTable from './components/MetricsTable.jsx'
import MetricsChart from './components/MetricsChart.jsx'
import ProbabilityChart from './components/ProbabilityChart.jsx'
import ConfusionMatrix from './components/ConfusionMatrix.jsx'
import PerClassMetrics from './components/PerClassMetrics.jsx'
import ClassDistribution from './components/ClassDistribution.jsx'
import Rachs1Comparison from './components/Rachs1Comparison.jsx'
import PredictionTab from './components/PredictionTab.jsx'
import styles from './App.module.css'

const POLL_INTERVAL = 3000

export default function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [pipelineStatus, setPipelineStatus] = useState({ running: false, last_status: 'idle' })
  const [activeModel, setActiveModel] = useState(null)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('dashboard')

  const fetchResults = useCallback(async () => {
    try {
      const res = await axios.get('/api/results')
      setData(res.data)
      setError(null)
      if (res.data?.models?.length > 0 && !activeModel) {
        const best = res.data.best_model
        if (best) {
          setActiveModel(`${best.model}__${best.dataset_type}`)
        } else {
          const first = res.data.models[0]
          setActiveModel(`${first.model}__${first.dataset_type}`)
        }
      }
    } catch {
      setError('No se pudo conectar con el servidor API.')
    } finally {
      setLoading(false)
    }
  }, [activeModel])

  const fetchStatus = useCallback(async () => {
    try {
      const res = await axios.get('/api/pipeline/status')
      setPipelineStatus(res.data)
      if (res.data.last_status === 'success' && !res.data.running) {
        fetchResults()
      }
    } catch {}
  }, [fetchResults])

  useEffect(() => {
    fetchResults()
    fetchStatus()
    const interval = setInterval(() => {
      fetchStatus()
    }, POLL_INTERVAL)
    return () => clearInterval(interval)
  }, [])

  const selectedModel = data?.models?.find(
    m => `${m.model}__${m.dataset_type}` === activeModel
  )

  return (
    <div className={styles.app}>
      <Header activeTab={activeTab} onTabChange={setActiveTab} />

      <main className={styles.main}>
        {activeTab === 'predict' ? (
          <PredictionTab />
        ) : (
          <>
            <PipelineControl
              status={pipelineStatus}
              onRefresh={fetchResults}
            />

            {loading && (
              <div className={styles.centered}>
                <div className={styles.spinner} />
                <p>Cargando datos...</p>
              </div>
            )}

            {error && (
              <div className={styles.errorBanner}>
                {error}
              </div>
            )}

            {!loading && data?.status === 'no_results' && (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>🔬</div>
                <h2>Sin resultados aún</h2>
                <p>Ejecuta el pipeline completo para ver las métricas de los modelos.</p>
                <p className={styles.hint}>Haz clic en "Ejecutar Pipeline" arriba para comenzar.</p>
              </div>
            )}

            {!loading && data?.status === 'ok' && (
              <>
                {data.best_model && (
                  <BestModelCard best={data.best_model} classes={data.target_classes} />
                )}

                {data.class_distribution && (
                  <ClassDistribution distribution={data.class_distribution} classes={data.target_classes} />
                )}

                <section className={styles.section}>
                  <h2 className={styles.sectionTitle}>Comparación de Modelos</h2>
                  <MetricsTable
                    rows={data.comparison_table}
                    bestModel={data.best_model}
                    activeModel={activeModel}
                    onSelect={setActiveModel}
                  />
                </section>

                <section className={styles.section}>
                  <h2 className={styles.sectionTitle}>Métricas por Modelo</h2>
                  <MetricsChart models={data.models} />
                </section>

                {selectedModel && (
                  <>
                    <section className={styles.section}>
                      <div className={styles.sectionHeader}>
                        <h2 className={styles.sectionTitle}>Detalle del Modelo Seleccionado</h2>
                        <div className={styles.modelSelector}>
                          {data.models.map(m => (
                            <button
                              key={`${m.model}__${m.dataset_type}`}
                              className={`${styles.modelBtn} ${activeModel === `${m.model}__${m.dataset_type}` ? styles.active : ''}`}
                              onClick={() => setActiveModel(`${m.model}__${m.dataset_type}`)}
                            >
                              {m.model.replace(/_/g, ' ')} [{m.dataset_type}]
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className={styles.detailGrid}>
                        <ProbabilityChart model={selectedModel} classes={data.target_classes} />
                        <TopFeaturesCard featureImportance={selectedModel.feature_importance} />
                      </div>
                      {selectedModel.confusion_matrix?.length > 0 && (
                        <div className={styles.confusionRow}>
                          <ConfusionMatrix
                            matrix={selectedModel.confusion_matrix}
                            classes={data.target_classes}
                          />
                        </div>
                      )}
                      <PerClassMetrics
                        rows={selectedModel.per_class_metrics}
                        classes={data.target_classes}
                        title={`Métricas por Clase — ${selectedModel.model.replace(/_/g, ' ')} [${selectedModel.dataset_type}]`}
                      />
                    </section>
                  </>
                )}

                <Rachs1Comparison
                  rachs1={data.rachs1}
                  bestModel={data.best_model}
                  models={data.models}
                />
              </>
            )}
          </>
        )}
      </main>
    </div>
  )
}
