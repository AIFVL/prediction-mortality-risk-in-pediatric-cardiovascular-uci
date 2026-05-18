import { useState } from 'react'
import axios from 'axios'
import styles from './PipelineControl.module.css'

const STATUS_COLORS = {
  idle: '#94a3b8',
  running: '#38bdf8',
  success: '#34d399',
  error: '#f87171',
}

const STATUS_LABELS = {
  idle: 'Sin ejecutar',
  running: 'Ejecutando...',
  success: 'Completado',
  error: 'Error',
}

export default function PipelineControl({ status, onRefresh }) {
  const [log, setLog] = useState([])
  const [showLog, setShowLog] = useState(false)
  const [running, setRunning] = useState(false)

  const handleRun = async () => {
    try {
      setRunning(true)
      // steps omitted → API uses default (includes feature_importance, rachs1)
      await axios.post('/api/pipeline/run', {})
    } catch (e) {
      if (e.response?.status === 409) {
        alert('El pipeline ya está en ejecución.')
      } else {
        alert('Error al iniciar el pipeline: ' + (e.response?.data?.detail || e.message))
      }
    } finally {
      setRunning(false)
    }
  }

  const handleViewLog = async () => {
    try {
      const res = await axios.get('/api/pipeline/log')
      setLog(res.data.log || [])
      setShowLog(true)
    } catch {}
  }

  const color = STATUS_COLORS[status.last_status] || STATUS_COLORS.idle
  const label = STATUS_LABELS[status.last_status] || 'Desconocido'

  return (
    <div className={styles.card}>
      <div className={styles.left}>
        <div className={styles.statusDot} style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
        <div>
          <span className={styles.statusLabel}>{label}</span>
          {status.last_run && (
            <span className={styles.lastRun}>Última ejecución: {status.last_run}</span>
          )}
        </div>
      </div>
      <div className={styles.actions}>
        <button className={styles.btnSecondary} onClick={onRefresh}>
          Actualizar
        </button>
        {status.last_run && (
          <button className={styles.btnSecondary} onClick={handleViewLog}>
            Ver Log
          </button>
        )}
        <button
          className={styles.btnPrimary}
          onClick={handleRun}
          disabled={status.running || running}
        >
          {status.running || running ? (
            <>
              <span className={styles.spinner} />
              Ejecutando...
            </>
          ) : 'Ejecutar Pipeline'}
        </button>
      </div>

      {showLog && (
        <div className={styles.logOverlay} onClick={() => setShowLog(false)}>
          <div className={styles.logModal} onClick={e => e.stopPropagation()}>
            <div className={styles.logHeader}>
              <h3>Log del Pipeline</h3>
              <button onClick={() => setShowLog(false)}>✕</button>
            </div>
            <pre className={styles.logContent}>
              {log.length > 0 ? log.join('\n') : 'Sin logs disponibles.'}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
