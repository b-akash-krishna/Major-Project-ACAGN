import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity, AlertCircle, CheckCircle, ChevronRight,
  Clock, FileText, Heart, RefreshCcw, Stethoscope,
  TrendingUp, User, Database, ArrowRight, Zap
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

/* ── Disease-specific advanced fields ── */
const DISEASE_FEATURES = {
  "General Medicine": [
    { name: "bmi", label: "BMI", step: "0.1", default: 24.5 },
    { name: "ed_time_hours", label: "ER Wait (hrs)", step: "0.1", default: 4.0 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 50 },
    { name: "lab_abnormal_count", label: "Abnormal Labs", step: "1", default: 10 },
    { name: "had_icu", label: "ICU Stays", step: "1", default: 0 },
  ],
  "Cardiology (CHF/Afib)": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 2.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 10.0 },
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "lab_chloride_max", label: "Max Chloride", step: "0.1", default: 105 },
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 20 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 80 },
  ],
  "Pulmonary (COPD/Pneumonia)": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 1.0 },
    { name: "lab_wbc_last", label: "WBC Count (Last)", step: "0.1", default: 12 },
    { name: "lab_hemoglobin_min", label: "Min Hemoglobin", step: "0.1", default: 10 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 60 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 2 },
    { name: "bmi", label: "BMI", step: "0.1", default: 28.5 },
  ],
  "Sepsis/Infection": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 5.0 },
    { name: "lab_wbc_min", label: "Min WBC Count", step: "0.1", default: 4 },
    { name: "lab_wbc_last", label: "WBC Count (Last)", step: "0.1", default: 20 },
    { name: "lab_platelets_min", label: "Min Platelets", step: "1", default: 150 },
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 30 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 120 },
  ],
  "Renal Failure": [
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 50 },
    { name: "lab_chloride_max", label: "Max Chloride", step: "0.1", default: 110 },
    { name: "anemia", label: "Anemia (0/1)", step: "1", default: 1 },
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 3.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 12.0 },
  ],
  "Diabetes/Endocrine": [
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 18 },
    { name: "bmi", label: "BMI", step: "0.1", default: 30.5 },
    { name: "lab_abnormal_count", label: "Abnormal Labs", step: "1", default: 15 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 80 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 8.0 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
  ],
  "Neurology (Stroke/Dementia)": [
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "lab_platelets_last", label: "Last Platelets", step: "1", default: 200 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 40 },
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 4.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 15.0 },
  ],
};

/* ── Patient Presets ── */
const PATIENT_PRESETS = [
  {
    // LOW risk — healthy young patient, first admission
    id: 'p01', name: 'Emma Thompson', desc: '22 yr · Appendectomy · LOW',
    data: {
      primary_disease: 'General Medicine',
      anchor_age: 22, gender: 0, los_days: 1.0,
      prev_admissions: 0, admission_type: 3,
      dx_count: 1, proc_count: 1, rx_count: 5,
      lab_abnormal_count: 0, lab_abnormal_rate: 0.0,
      had_icu: 0, days_since_last: 3650, prev_readmit_rate: 0.0,
      bmi: 21.0, ed_time_hours: 1.5, transfer_count: 0, med_admin_count: 5,
      note: "22-year-old female admitted for elective appendectomy. No complications. Good general health.",
    },
  },
  {
    // MEDIUM risk — CHF patient, 5 prior admits, 60 days since last
    id: 'p06', name: 'Dorothy Chen', desc: '70 yr · CHF Decompensation · MEDIUM',
    data: {
      primary_disease: 'Cardiology (CHF/Afib)',
      anchor_age: 70, gender: 0, los_days: 9.0,
      prev_admissions: 5, admission_type: 1,
      dx_count: 8, proc_count: 6, rx_count: 100,
      lab_abnormal_count: 45, lab_abnormal_rate: 0.50,
      had_icu: 1, days_since_last: 60, prev_readmit_rate: 0.45,
      icu_los_sum: 3.0, prev_los_mean: 9.0,
      hyponatremia: 1, lab_chloride_max: 109, lab_bun_last: 32,
      med_admin_count: 200,
      note: "70-year-old obese female with CHF and afib. Acute decompensation with fluid overload. 5 prior admissions in 2 years. On IV diuretics and vasodilators.",
    },
  },
  {
    // HIGH risk — ESRD + sepsis, 12 prior admits, 10 days since last, extreme labs
    id: 'p09', name: 'George Williams', desc: '68 yr · ESRD + Sepsis · HIGH',
    data: {
      primary_disease: 'Sepsis/Infection',
      anchor_age: 68, gender: 1, los_days: 20.0,
      prev_admissions: 12, admission_type: 1,
      dx_count: 18, proc_count: 25, rx_count: 300,
      lab_abnormal_count: 200, lab_abnormal_rate: 0.85,
      had_icu: 2, days_since_last: 10, prev_readmit_rate: 0.95,
      icu_los_sum: 12.0, prev_los_mean: 15.0,
      lab_wbc_min: 2.1, lab_wbc_last: 28.0, lab_platelets_min: 40, lab_bun_last: 85,
      med_admin_count: 450,
      note: "68-year-old male with ESRD on dialysis, diabetic foot infection progressing to sepsis. Blood cultures positive. Frequent hospitalizations last 2 years. ICU admission.",
    },
  },
];

/* ── Risk colours / labels ── */
const RISK_META = {
  HIGH:   { color: '#c0392b', label: 'Priority Wellness Opportunity', alertClass: 'high', icon: AlertCircle },
  MEDIUM: { color: '#c8922a', label: 'Moderate Attention Advised',     alertClass: '',    icon: TrendingUp },
  LOW:    { color: '#1e7a5c', label: 'Vitality Stability Confirmed',   alertClass: 'low', icon: CheckCircle },
};

const CIRCUMFERENCE = 2 * Math.PI * 80; // r=80

export default function App() {
  const [formData, setFormData] = useState({
    ...PATIENT_PRESETS[0].data,
    primary_disease: 'General Medicine',
  });
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [activeTab, setActiveTab]       = useState('prediction');
  const [showAdvanced, setShowAdvanced] = useState(false);

  /* ── Handlers ── */
  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name === 'note' || name === 'primary_disease') {
      setFormData(p => ({ ...p, [name]: value }));
      return;
    }
    const isFloat = e.target.step && e.target.step.includes('.');
    setFormData(p => ({ ...p, [name]: isFloat ? parseFloat(value) || 0 : parseInt(value) || 0 }));
  };

  const loadPreset = (preset) => { setFormData(preset.data); setResult(null); };

  const runPrediction = async () => {
    setLoading(true);
    try {
      const res = await axios.post(`${API_URL}/predict`, formData);
      setResult(res.data);
    } catch (err) {
      console.error(err);
      alert('Cannot reach prediction server. Ensure server.py is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  /* ── SVG Arc ── */
  const riskMeta  = result ? (RISK_META[result.risk] || RISK_META.LOW) : null;
  const arcOffset = result ? CIRCUMFERENCE * (1 - result.probability) : CIRCUMFERENCE;

  const advancedFields = DISEASE_FEATURES[formData.primary_disease] || DISEASE_FEATURES['General Medicine'];

  /* ── Render ── */
  return (
    <div className="app-shell">

      {/* ── Header ── */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-orb" />
          <div>
            <div className="brand-name">WeCare<span>.ai</span></div>
            <div className="brand-tagline">Longevity Intelligence Platform</div>
          </div>
        </div>

        <nav className="nav">
          {['Prediction', 'History', 'Patients', 'Settings'].map(t => (
            <button
              key={t}
              className={`nav-btn${activeTab === t.toLowerCase() ? ' active' : ''}`}
              onClick={() => setActiveTab(t.toLowerCase())}
              aria-current={activeTab === t.toLowerCase() ? 'page' : undefined}
            >{t}</button>
          ))}
        </nav>

        <div className="user-avatar" aria-label="Profile"><User size={15} /></div>
      </header>

      {/* ── Main ── */}
      <main className="app-main">

        {/* ── Sidebar ── */}
        <aside className="sidebar" aria-label="Assessment Form">

          <div className="sidebar-title">
            <Stethoscope size={18} className="sidebar-icon" aria-hidden />
            <h2>Patient Assessment</h2>
          </div>

          {/* Presets */}
          <p className="preset-section-label">Clinical Profiles</p>
          <div className="preset-list">
            {PATIENT_PRESETS.map(p => (
              <button key={p.id} className="preset-card" onClick={() => loadPreset(p)} aria-label={`Load ${p.name}`}>
                <div>
                  <div className="preset-card-name">{p.name}</div>
                  <div className="preset-card-desc">{p.desc}</div>
                </div>
                <ChevronRight size={15} className="preset-card-arrow" aria-hidden />
              </button>
            ))}
          </div>

          <hr className="divider" />
          <p className="section-label">Core Metrics</p>

          {/* Disease */}
          <div className="input-group">
            <label className="input-label">Primary Pathology</label>
            <select name="primary_disease" value={formData.primary_disease} onChange={handleChange} className="input-field">
              {Object.keys(DISEASE_FEATURES).map(d => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Age + Gender */}
          <div className="grid-2">
            <div className="input-group">
              <label className="input-label">Age</label>
              <input type="number" name="anchor_age" value={formData.anchor_age} onChange={handleChange} className="input-field" />
            </div>
            <div className="input-group">
              <label className="input-label">Gender</label>
              <select name="gender" value={formData.gender} onChange={handleChange} className="input-field">
                <option value={0}>Female</option>
                <option value={1}>Male</option>
              </select>
            </div>
          </div>

          {/* LOS */}
          <div className="input-group">
            <label className="input-label">Length of Stay (days)</label>
            <input type="number" step="0.5" name="los_days" value={formData.los_days} onChange={handleChange} className="input-field" />
          </div>

          {/* Prev Admissions */}
          <div className="input-group">
            <label className="input-label">Previous Admissions</label>
            <input type="number" name="prev_admissions" value={formData.prev_admissions} onChange={handleChange} className="input-field" />
          </div>

          {/* Dx + Rx */}
          <div className="grid-2">
            <div className="input-group">
              <label className="input-label">Diagnoses</label>
              <input type="number" name="dx_count" value={formData.dx_count} onChange={handleChange} className="input-field" />
            </div>
            <div className="input-group">
              <label className="input-label">Medications</label>
              <input type="number" name="rx_count" value={formData.rx_count} onChange={handleChange} className="input-field" />
            </div>
          </div>

          {/* Lab Rate + Days Since Last */}
          <div className="grid-2">
            <div className="input-group">
              <label className="input-label">Lab Abnormal Rate</label>
              <input type="number" step="0.01" min="0" max="1" name="lab_abnormal_rate" value={formData.lab_abnormal_rate} onChange={handleChange} className="input-field" />
            </div>
            <div className="input-group">
              <label className="input-label">Days Since Last</label>
              <input type="number" name="days_since_last" value={formData.days_since_last} onChange={handleChange} className="input-field" />
            </div>
          </div>

          {/* Prior Readmit Rate */}
          <div className="input-group">
            <label className="input-label">Prior Readmission Rate (0–1)</label>
            <input type="number" step="0.01" min="0" max="1" name="prev_readmit_rate" value={formData.prev_readmit_rate} onChange={handleChange} className="input-field" />
          </div>

          {/* Advanced Toggle */}
          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(v => !v)}
            aria-expanded={showAdvanced}
          >
            <span>Disease-Specific Parameters</span>
            <ChevronRight size={14} className={`advanced-toggle-icon${showAdvanced ? ' open' : ''}`} aria-hidden />
          </button>

          <AnimatePresence mode="wait">
            {showAdvanced && (
              <motion.div
                key={formData.primary_disease}
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: 'easeInOut' }}
                className="overflow-hidden"
                style={{ marginBottom: '1rem' }}
              >
                <div className="grid-2">
                  {advancedFields.map(f => (
                    <div key={f.name} className="input-group">
                      <label className="input-label">{f.label}</label>
                      <input
                        type="number"
                        step={f.step}
                        name={f.name}
                        value={formData[f.name] ?? f.default}
                        onChange={handleChange}
                        className="input-field"
                      />
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Clinical Note */}
          <div className="input-group">
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:'0.4rem' }}>
              <label className="input-label" style={{ margin:0 }}>Physician Note</label>
              <span style={{ fontSize:'0.6rem', letterSpacing:'0.1em', textTransform:'uppercase', color:'var(--gold)', border:'1px solid var(--border)', borderRadius:'4px', padding:'0.1rem 0.4rem' }}>NLP</span>
            </div>
            <textarea
              name="note"
              rows={4}
              value={formData.note || ''}
              onChange={handleChange}
              className="note-field"
              placeholder="Paste discharge summary or clinical notes…"
            />
          </div>

          <button className="btn-analyze" onClick={runPrediction} disabled={loading}>
            {loading
              ? <><RefreshCcw size={16} className="spin" /> Analysing…</>
              : <><Zap size={16} /> Generate Vitality Forecast</>
            }
          </button>
        </aside>

        {/* ── Results Panel ── */}
        <section className="results-panel" aria-live="polite">
          <AnimatePresence mode="wait">
            {!result ? (
              <motion.div
                key="empty"
                className="empty-state"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <div className="vitality-orb-wrap">
                  <div className="orb-ring-2" />
                  <div className="orb-ring" />
                  <div className="vitality-orb">
                    <div className="orb-inner" />
                  </div>
                </div>

                <h3 className="empty-title">Awaiting Your Assessment</h3>
                <p className="empty-subtitle">
                  Select a clinical profile from the sidebar or enter patient data to generate a
                  real-time vitality forecast using the TRANCE intelligence framework.
                </p>
                <button className="btn-start" onClick={runPrediction}>
                  Initiate Forecast <ArrowRight size={16} />
                </button>
              </motion.div>
            ) : (
              <motion.div
                key={result.probability}
                className="result-grid"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              >
                {/* ── Score Hero ── */}
                <div className="score-hero">
                  <div className="score-orb-wrap">
                    <svg className="score-svg" viewBox="0 0 180 180">
                      <circle className="score-track" cx="90" cy="90" r="80" />
                      <motion.circle
                        className="score-arc"
                        cx="90" cy="90" r="80"
                        stroke={riskMeta.color}
                        strokeDasharray={CIRCUMFERENCE}
                        initial={{ strokeDashoffset: CIRCUMFERENCE }}
                        animate={{ strokeDashoffset: arcOffset }}
                        transition={{ duration: 1.6, ease: 'easeOut' }}
                        strokeLinecap="round"
                        fill="none"
                        strokeWidth="8"
                      />
                    </svg>
                    <div className="score-center">
                      <span className="score-pct" style={{ color: riskMeta.color }}>
                        {(result.probability * 100).toFixed(1)}%
                      </span>
                      <span className="score-label">Risk Score</span>
                    </div>
                  </div>

                  <div className="score-meta">
                    <div className={`risk-badge ${result.risk}`}>{result.risk}</div>
                    <h2 className="score-heading">Vitality Forecast</h2>
                    <p className="score-sub">
                      {riskMeta.label}. The TRANCE model has analysed <strong style={{color:'var(--oyster)'}}>{formData.dx_count}</strong> diagnoses
                      and <strong style={{color:'var(--oyster)'}}>physician notes</strong> to determine the 30-day readmission likelihood for this patient.
                    </p>
                    {result.risk === 'HIGH' && (
                      <div className="risk-alert high">
                        <AlertCircle size={14} /> Immediate clinical review recommended
                      </div>
                    )}
                    {result.risk === 'LOW' && (
                      <div className="risk-alert low">
                        <CheckCircle size={14} /> Patient demonstrates strong recovery indicators
                      </div>
                    )}
                  </div>
                </div>

                {/* ── Stats Row ── */}
                <div className="stats-row">
                  <div className="stat-card">
                    <div className="stat-icon gold"><Clock size={16} /></div>
                    <div className="stat-name">Stay Duration</div>
                    <div className="stat-value">{formData.los_days}<span style={{fontSize:'1rem',color:'var(--oyster-dim)'}}> d</span></div>
                    <div className="stat-bar">
                      <div className="stat-bar-fill gold" style={{ width: `${Math.min(100, (formData.los_days / 30) * 100)}%` }} />
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-icon emerald"><Activity size={16} /></div>
                    <div className="stat-name">Abnormal Labs</div>
                    <div className="stat-value">{formData.lab_abnormal_count}<span style={{fontSize:'1rem',color:'var(--oyster-dim)'}}> hits</span></div>
                    <div className="stat-sub">{(formData.lab_abnormal_rate * 100).toFixed(0)}% abnormal rate</div>
                    <div className="stat-bar">
                      <div className="stat-bar-fill emerald" style={{ width: `${Math.min(100, formData.lab_abnormal_rate * 100)}%` }} />
                    </div>
                  </div>

                  <div className="stat-card">
                    <div className="stat-icon silver"><RefreshCcw size={16} /></div>
                    <div className="stat-name">Admission History</div>
                    <div className="stat-value">{formData.prev_admissions}<span style={{fontSize:'1rem',color:'var(--oyster-dim)'}}> prior</span></div>
                    <div className="stat-sub">{formData.days_since_last} days since last visit</div>
                  </div>
                </div>

                {/* ── Clinical Note ── */}
                {formData.note && (
                  <div className="note-card">
                    <div className="card-header">
                      <div className="card-title"><FileText size={14} /> Physician Note Analysis</div>
                      <span className="card-tag">Natural Language Processing</span>
                    </div>
                    <p className="note-text">{formData.note}</p>
                  </div>
                )}

                {/* ── Insights Row ── */}
                <div className="insights-row">
                  <div className="insight-card">
                    <div className="card-header">
                      <div className="card-title"><Database size={14} /> Intelligent Imputation</div>
                    </div>
                    <p>
                      The model has anchored this forecast to a <strong>Statistical Patient Profile</strong> of 160+ parameters
                      calibrated to <strong>{result?.stats?.disease_context || 'General Medicine'}</strong> patterns from 546,000+ MIMIC-III records.
                    </p>
                    <div className="completeness-bar">
                      <div className="completeness-fill" style={{ width: `${(result?.stats?.completeness || 0) * 100}%` }} />
                    </div>
                    <div className="completeness-label">
                      <span>Data Completeness</span>
                      <span className="val">{((result?.stats?.completeness || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>

                  <div className="insight-card">
                    <div className="card-header">
                      <div className="card-title"><Heart size={14} /> TRANCE Framework</div>
                    </div>
                    <p>
                      Powered by the <strong>Transformer-Augmented Neural Clinical Engine</strong>, trained on real-world EHR data.
                      Disease-specific ICD markers and <strong>512-dimensional NLP embeddings</strong> ensure precision per pathology.
                    </p>
                    <div style={{ marginTop:'0.75rem', display:'flex', gap:'0.5rem' }}>
                      <span style={{ fontSize:'0.65rem', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--gold)', border:'1px solid var(--border)', borderRadius:'4px', padding:'0.15rem 0.5rem' }}>
                        {result?.stats?.features_used} Features
                      </span>
                      <span style={{ fontSize:'0.65rem', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--oyster-dim)', border:'1px solid var(--border-subtle)', borderRadius:'4px', padding:'0.15rem 0.5rem' }}>
                        v2.4.1 Stable
                      </span>
                    </div>
                  </div>
                </div>

              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>

      {/* ── Footer ── */}
      <footer className="app-footer">
        <div style={{ display:'flex', gap:'1.5rem' }}>
          <span>© 2026 WeCare AI Health Systems</span>
          <span>Privacy Policy</span>
          <span>HIPAA Compliant</span>
        </div>
        <div className="footer-status">
          <div className="status-dot" />
          <span>Model Engine v2.4.1 · Active</span>
        </div>
      </footer>

    </div>
  );
}
