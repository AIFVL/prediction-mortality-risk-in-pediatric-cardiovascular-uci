import styles from './Header.module.css'

export default function Header({ activeTab, onTabChange }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <div className={styles.logo}>
          <span className={styles.icon}>🫀</span>
          <div>
            <h1 className={styles.title}>UCI Cardiovascular Pediátrica</h1>
            <p className={styles.subtitle}>Dashboard de Predicción de Mortalidad</p>
          </div>
        </div>

        <nav className={styles.nav}>
          <button
            className={`${styles.navBtn} ${activeTab === 'dashboard' ? styles.navActive : ''}`}
            onClick={() => onTabChange('dashboard')}
          >
            Tablero
          </button>
          <button
            className={`${styles.navBtn} ${activeTab === 'predict' ? styles.navActive : ''}`}
            onClick={() => onTabChange('predict')}
          >
            Predicción
          </button>
        </nav>

        <div className={styles.badge}>ML Pipeline</div>
      </div>
    </header>
  )
}
