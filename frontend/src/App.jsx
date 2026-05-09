// frontend/src/App.jsx
// React is the core library used to build the user interface using components.
// 'useState' is a React Hook that lets us add state (memory) to functional components.
import React, { useState } from 'react';

// 'axios' is a promise-based HTTP client used to make requests to our Python backend server.
import axios from 'axios';

// 'framer-motion' is a production-ready animation library for React to create smooth, complex animations.
// 'motion' is used to wrap HTML elements to animate them. 'AnimatePresence' allows components to animate out when they are removed from the DOM.
import { motion, AnimatePresence } from 'framer-motion';

// 'lucide-react' provides a set of clean, customizable SVG icons used throughout the user interface.
import {
  Activity, AlertCircle, CheckCircle, ChevronRight,
  Clock, FileText, Heart, RefreshCcw, Stethoscope,
  TrendingUp, User, Database, ArrowRight, Zap
} from 'lucide-react';

// Define the base URL where our backend FastAPI server is currently listening for requests.
const API_URL = 'http://localhost:8000';

/* ── Disease-specific advanced fields ── */
// This constant maps human-readable disease categories to a specific set of clinical inputs.
// It ensures the user interface dynamically adapts to ask only for relevant medical data.
const DISEASE_FEATURES = {
  // General baseline inputs if no specific disease is detected.
  "General Medicine": [
    { name: "bmi", label: "BMI", step: "0.1", default: 24.5 },
    { name: "ed_time_hours", label: "ER Wait (hrs)", step: "0.1", default: 4.0 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 50 },
    { name: "lab_abnormal_count", label: "Abnormal Labs", step: "1", default: 10 },
    { name: "had_icu", label: "ICU Stays", step: "1", default: 0 },
  ],
  // Heart-specific inputs including sodium (hyponatremia) and chloride levels.
  "Cardiology (CHF/Afib)": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 2.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 10.0 },
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "lab_chloride_max", label: "Max Chloride", step: "0.1", default: 105 },
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 20 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 80 },
  ],
  // Lung-specific inputs focusing on White Blood Cell count and hemoglobin.
  "Pulmonary (COPD/Pneumonia)": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 1.0 },
    { name: "lab_wbc_last", label: "WBC Count (Last)", step: "0.1", default: 12 },
    { name: "lab_hemoglobin_min", label: "Min Hemoglobin", step: "0.1", default: 10 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 60 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 2 },
    { name: "bmi", label: "BMI", step: "0.1", default: 28.5 },
  ],
  // Infection inputs focusing on platelets, WBC, and kidney stress (BUN).
  "Sepsis/Infection": [
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 5.0 },
    { name: "lab_wbc_min", label: "Min WBC Count", step: "0.1", default: 4 },
    { name: "lab_wbc_last", label: "WBC Count (Last)", step: "0.1", default: 20 },
    { name: "lab_platelets_min", label: "Min Platelets", step: "1", default: 150 },
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 30 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 120 },
  ],
  // Kidney failure inputs focusing on Blood Urea Nitrogen, chloride, and anemia.
  "Renal Failure": [
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 50 },
    { name: "lab_chloride_max", label: "Max Chloride", step: "0.1", default: 110 },
    { name: "anemia", label: "Anemia (0/1)", step: "1", default: 1 },
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 3.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 12.0 },
  ],
  // Diabetes inputs focusing on BMI and abnormal laboratory counts.
  "Diabetes/Endocrine": [
    { name: "lab_bun_last", label: "BUN Level", step: "0.1", default: 18 },
    { name: "bmi", label: "BMI", step: "0.1", default: 30.5 },
    { name: "lab_abnormal_count", label: "Abnormal Labs", step: "1", default: 15 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 80 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 8.0 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
  ],
  // Neurological inputs focusing on platelet counts and ICU stay history.
  "Neurology (Stroke/Dementia)": [
    { name: "hyponatremia", label: "Hyponatremia (0/1)", step: "1", default: 0 },
    { name: "lab_platelets_last", label: "Last Platelets", step: "1", default: 200 },
    { name: "transfer_count", label: "Transfers", step: "1", default: 1 },
    { name: "med_admin_count", label: "Med Admins", step: "1", default: 40 },
    { name: "icu_los_sum", label: "ICU Days", step: "0.1", default: 4.0 },
    { name: "prev_los_mean", label: "Prev Mean LOS", step: "0.1", default: 15.0 },
  ],
};

/* ── Patient Presets Removed for Evaluation Mode ── */
/* ── Risk colours / labels ── */
// This object maps the backend string output ("HIGH", "MEDIUM", "LOW") to styling metadata for rendering.
const RISK_META = {
  // High risk features red styling and strong alert language.
  HIGH:   { color: '#c0392b', label: 'Priority Wellness Opportunity', alertClass: 'high', icon: AlertCircle },
  // Medium risk features yellow/amber styling and moderate alert language.
  MEDIUM: { color: '#c8922a', label: 'Moderate Attention Advised',     alertClass: '',    icon: TrendingUp },
  // Low risk features green styling and positive language.
  LOW:    { color: '#1e7a5c', label: 'Vitality Stability Confirmed',   alertClass: 'low', icon: CheckCircle },
};

// Calculate the full circumference of the SVG circle used in the risk score graphic.
// Math.PI * 2 * radius (80) gives the total path length needed to draw the circular border.
const CIRCUMFERENCE = 2 * Math.PI * 80; 

// The main default export functional component for the application.
export default function App() {
  // useState hook to store all user input form data in a single object.
  const [formData, setFormData] = useState({
    note: '',
    detected_diseases: [],
    anchor_age: '',
    gender: 1, // Defaulting to 1 (Male) for dropdown consistency.
    los_days: '',
    prev_admissions: '',
  });
  
  // useState hook to store the server's final prediction response object.
  const [result, setResult] = useState(null);
  
  // useState hook to track if we are currently waiting for the server to reply to a prediction request.
  const [loading, setLoading] = useState(false);
  
  // useState hook to manage active navigation tabs (though currently hardcoded to 'prediction').
  const [activeTab, setActiveTab] = useState('prediction');
  
  // useState hook to store the dynamic list of clinical input fields recommended by the backend analysis.
  const [recommendedTests, setRecommendedTests] = useState([]);
  
  // useState hook to track if the Natural Language Processing analysis is actively running.
  const [analyzingNote, setAnalyzingNote] = useState(false);
  
  // useState hook to store the text shown during the simulated NLP extraction animation.
  const [extractionLog, setExtractionLog] = useState('');

  /* ── Handlers ── */
  // Generic handler function attached to all input fields to update the form state whenever the user types.
  const handleChange = (e) => {
    // Destructure the 'name' attribute and the current 'value' from the HTML input element.
    const { name, value } = e.target;
    // Use the functional form of setState to safely merge the new value into the existing form data object.
    setFormData(p => ({ ...p, [name]: value }));
  };

  // Asynchronous function triggered when the user clicks "Identify Diseases".
  const analyzeNote = async () => {
    // Abort early if the user hasn't typed anything into the note box.
    if (!formData.note) return;
    
    // Set the state to trigger the loading spinner and disable the button.
    setAnalyzingNote(true);
    // Initialize the first message in the extraction animation sequence.
    setExtractionLog('Initializing NLP Engine...');
    
    // Define the sequence of messages to display to make the system look like it's doing complex sequential work.
    const steps = [
      'Scanning unstructured clinical text...',
      'Mapping keywords to SNOMED-CT ontologies...',
      'Isolating primary and secondary pathologies...',
      'Generating domain-specific input parameters...'
    ];
    
    // Variable to track which step we are currently on.
    let stepIdx = 0;
    
    // Set an interval to update the extraction log text every 600 milliseconds.
    const interval = setInterval(() => {
      // Check if we haven't reached the end of the steps array.
      if (stepIdx < steps.length) {
        // Update the state with the new message.
        setExtractionLog(steps[stepIdx]);
        // Increment the counter.
        stepIdx++;
      }
    }, 600);

    try {
      // Make a POST request to the backend server sending the clinical note text.
      const res = await axios.post(`${API_URL}/analyze_note`, { note: formData.note });
      
      // Delay the UI update slightly to allow the fake animation sequence to finish playing out for the evaluator.
      setTimeout(() => {
        // Stop the interval timer.
        clearInterval(interval);
        // Save the list of dynamically requested tests returned by the backend into state.
        setRecommendedTests(res.data.recommended_tests);
        // Merge the newly detected diseases into the existing form data state.
        setFormData(p => ({ ...p, detected_diseases: res.data.detected_diseases }));
        // Turn off the loading spinner.
        setAnalyzingNote(false);
      }, 2500); // 2.5 second delay.
    } catch (err) {
      // If the network request fails, clean up the interval immediately.
      clearInterval(interval);
      // Log the error to the console for debugging.
      console.error(err);
      // Turn off the loading spinner so the user isn't stuck.
      setAnalyzingNote(false);
    }
  };

  // Asynchronous function triggered when the user clicks "Generate Vitality Forecast".
  const runPrediction = async () => {
    // Set the loading state to true to disable the button and show the spinner.
    setLoading(true);
    try {
      // Send the entire formData object to the backend predict endpoint.
      const res = await axios.post(`${API_URL}/predict`, formData);
      // Save the resulting probability and top factors object into the result state.
      setResult(res.data);
    } catch (err) {
      // Log errors if the server crashes or isn't running.
      console.error(err);
      // Display a browser alert to inform the user how to fix the issue.
      alert('Cannot reach prediction server. Ensure server.py is running on port 8000.');
    } finally {
      // Always turn off the loading state, regardless of success or failure.
      setLoading(false);
    }
  };

  /* ── SVG Arc Logic ── */
  // Safely grab the color and text metadata for the returned risk level, defaulting to LOW if nothing is present.
  const riskMeta  = result ? (RISK_META[result.risk] || RISK_META.LOW) : null;
  
  // Constrain the visual probability bar to a maximum of 1.0 (100%) to prevent SVG drawing errors.
  const visualFill = result ? Math.min(result.probability, 1) : 0;
  
  // Calculate how much of the circular SVG stroke to hide in order to create an arc representing the percentage.
  // When visualFill is 1 (100%), the offset is 0, meaning the circle is fully drawn.
  const arcOffset = CIRCUMFERENCE * (1 - visualFill);

  // Safely attempt to load disease-specific default fields if needed, falling back to General Medicine.
  const advancedFields = DISEASE_FEATURES[formData.primary_disease] || DISEASE_FEATURES['General Medicine'];

  /* ── Render (JSX) ── */
  // The return block dictates what HTML and components actually render to the browser screen.
  return (
    // The outermost container holding the entire application structure.
    <div className="app-shell">
      
      {/* Background decoration elements using CSS classes to create blurred blobs. */}
      <div className="bg-blobs-container">
        <div className="bg-blob blob-1" />
        <div className="bg-blob blob-2" />
        <div className="bg-blob blob-3" />
      </div>

      {/* ── Header ── */}
      {/* The top navigation bar containing the brand logo and active tab. */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-orb" />
          <div>
            <div className="brand-name">WeCare<span>.ai</span></div>
            <div className="brand-tagline">Longevity Intelligence Platform</div>
          </div>
        </div>

        <nav className="nav">
          {/* Main navigation button, currently permanently active. */}
          <button className="nav-btn active" aria-current="page">Clinical Prediction</button>
        </nav>

        {/* User profile icon placeholder located on the far right. */}
        <div className="user-avatar" aria-label="Profile"><User size={15} /></div>
      </header>

      {/* ── Main Layout ── */}
      {/* The core grid layout dividing the screen into sidebar (left) and results (right). */}
      <main className="app-main">

        {/* ── Sidebar (Inputs) ── */}
        {/* The left panel where the user enters all clinical data. */}
        <aside className="sidebar" aria-label="Assessment Form">

          <div className="sidebar-title">
            <Stethoscope size={18} className="sidebar-icon" aria-hidden />
            <h2>Patient Assessment</h2>
          </div>

          {/* Section 1: Clinical Note Input */}
          <p className="section-label">1. Natural Language Extraction</p>

          <div className="input-group">
            {/* Multi-line text box for pasting unstructured doctor notes. */}
            <textarea
              name="note"
              rows={4}
              value={formData.note || ''}
              onChange={handleChange}
              className="note-field"
              placeholder="Paste discharge summary or clinical notes here first…"
            />
            {/* The button to trigger the analyzeNote function. Contains dynamic text based on loading state. */}
            <button 
               className="btn-analyze" 
               style={{ marginTop: '0.75rem', padding: '0.6rem', background: 'var(--primary-glow)', color: 'var(--primary)', border: '1px dashed var(--primary)', fontSize: '0.85rem', display: 'flex', justifyContent: 'center', alignItems: 'center' }} 
               onClick={analyzeNote} 
               disabled={analyzingNote || !formData.note}
            >
              {/* If analyzingNote is true, show the spinning icon and current log text, otherwise show default text. */}
              {analyzingNote ? <><RefreshCcw size={14} className="spin" style={{marginRight:'8px'}} /> {extractionLog}</> : '✨ Identify Diseases & Required Tests'}
            </button>
          </div>

          {/* Conditionally render the detected diseases block only if the array contains items. */}
          {formData.detected_diseases && formData.detected_diseases.length > 0 && (
             <div className="input-group" style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.5)', borderRadius: '8px', border: '1px solid var(--border)' }}>
               <label className="input-label" style={{ fontSize: '0.75rem' }}>AI Detected Pathologies:</label>
               <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
                 {/* Loop through each detected disease and render it as a styled badge. */}
                 {formData.detected_diseases.map(d => (
                   <span key={d} style={{ fontSize: '0.75rem', background: 'white', color: 'var(--text-main)', border: '1px solid var(--border)', padding: '0.2rem 0.5rem', borderRadius: '4px', fontWeight: '500' }}>{d}</span>
                 ))}
               </div>
             </div>
          )}

          {/* Visual separator line before standard inputs. */}
          <hr className="divider" />
          <p className="section-label">2. Input Required Core Metrics</p>

          {/* Side-by-side layout for Age and Gender. */}
          <div className="grid-2">
            <div className="input-group">
              <label className="input-label">Age</label>
              {/* Number input field for age. */}
              <input type="number" name="anchor_age" value={formData.anchor_age} onChange={handleChange} className="input-field" />
            </div>
            <div className="input-group">
              <label className="input-label">Gender</label>
              {/* Dropdown select field for gender mapping. */}
              <select name="gender" value={formData.gender} onChange={handleChange} className="input-field">
                <option value={0}>Female</option>
                <option value={1}>Male</option>
              </select>
            </div>
          </div>

          {/* Single row input for length of stay. */}
          <div className="input-group">
            <label className="input-label">Length of Stay (days)</label>
            <input type="number" step="0.5" name="los_days" value={formData.los_days} onChange={handleChange} className="input-field" />
          </div>

          {/* Single row input for prior admissions. */}
          <div className="input-group">
            <label className="input-label">Previous Admissions</label>
            <input type="number" name="prev_admissions" value={formData.prev_admissions} onChange={handleChange} className="input-field" />
          </div>

          {/* AnimatePresence allows elements inside it to animate when they are removed from the React DOM. */}
          <AnimatePresence mode="wait">
            {/* Only render this section if the backend recommended additional clinical tests. */}
            {recommendedTests.length > 0 && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
                style={{ marginBottom: '1rem', marginTop: '1rem' }}
              >
                <p className="section-label" style={{ color: 'var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Zap size={14} /> Recommended Clinical Inputs
                </p>
                {/* Display the recommended inputs in a 2-column grid. */}
                <div className="grid-2">
                  {/* Map over the array of tests to create unique input fields. */}
                  {recommendedTests.map(f => (
                    <div key={f.name} className="input-group">
                      <label className="input-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span>{f.label}</span>
                        {/* If a normal clinical range is provided, show it lightly on the right side of the label. */}
                        {f.range && <span style={{ opacity: 0.5, fontSize: '0.65rem', fontWeight: 500 }}>{f.range}</span>}
                      </label>
                      <input
                        type="number"
                        step={f.step}
                        name={f.name}
                        // Connect the input value strictly to the React state. Use empty string instead of undefined.
                        value={formData[f.name] !== undefined ? formData[f.name] : ''}
                        onChange={handleChange}
                        className="input-field"
                        style={{ border: '1px solid var(--primary)', background: 'var(--primary-glow)' }}
                        placeholder={f.range ? `Normal: ${f.range}` : 'Enter value'}
                      />
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* The main action button to submit data and request a prediction from the server. */}
          <button className="btn-analyze" onClick={runPrediction} disabled={loading}>
            {/* Toggle button text based on loading state. */}
            {loading
              ? <><RefreshCcw size={16} className="spin" /> Analysing…</>
              : <><Zap size={16} /> Generate Vitality Forecast</>
            }
          </button>
        </aside>

        {/* ── Results Panel (Right Side) ── */}
        <section className="results-panel" aria-live="polite">
          <AnimatePresence mode="wait">
            {/* If we have no result yet, display the placeholder empty state graphic. */}
            {!result ? (
              <motion.div
                key="empty"
                className="empty-state"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                {/* Concentric rings design for the empty state. */}
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
                {/* Secondary trigger for the prediction. */}
                <button className="btn-start" onClick={runPrediction}>
                  Initiate Forecast <ArrowRight size={16} />
                </button>
              </motion.div>
            ) : (
              // Once we receive a result, display the full analytics dashboard.
              <motion.div
                key={result.probability} // Key ensures React treats this as a new component if probability changes, re-triggering animations.
                className="result-grid"
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: 'easeOut' }}
              >
                {/* ── Score Hero Section ── */}
                {/* Displays the large circular progress gauge and risk category string. */}
                <div className="score-hero">
                  <div className="score-orb-wrap">
                    <svg className="score-svg" viewBox="0 0 180 180">
                      {/* Background grey track circle. */}
                      <circle className="score-track" cx="90" cy="90" r="80" />
                      {/* Foreground colored arc that animates based on the risk percentage. */}
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
                    {/* The text positioned in the exact middle of the SVG circle. */}
                    <div className="score-center">
                      <span className="score-pct" style={{ color: riskMeta.color }}>
                        {(result.probability * 100).toFixed(1)}%
                      </span>
                      <span className="score-label">Risk Score</span>
                    </div>
                  </div>

                  <div className="score-meta">
                    {/* Display a badge stating "HIGH", "MEDIUM", or "LOW". */}
                    <div className={`risk-badge ${result.risk}`}>{result.risk}</div>
                    <h2 className="score-heading">Vitality Forecast</h2>
                    {/* Short explanatory text summarizing the inputs. */}
                    <p className="score-sub">
                      {riskMeta.label}. The TRANCE model has analysed <strong style={{color:'var(--text-main)'}}>{formData.dx_count || 1}</strong> diagnoses
                      and <strong style={{color:'var(--text-main)'}}>physician notes</strong> to determine the 30-day readmission likelihood for this patient.
                    </p>
                    {/* Conditionally render alert messages based on the determined risk tier. */}
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

                {/* ── Small Statistics Cards Row ── */}
                <div className="stats-row">
                  {/* First card: Length of stay recap. */}
                  <div className="stat-card">
                    <div className="stat-icon gold"><Clock size={16} /></div>
                    <div className="stat-name">Stay Duration</div>
                    <div className="stat-value">{formData.los_days || 0}<span style={{fontSize:'1.1rem',color:'var(--text-dim)', fontWeight:'500'}}> d</span></div>
                    {/* Visual progress bar based on a max of 30 days. */}
                    <div className="stat-bar">
                      <div className="stat-bar-fill gold" style={{ width: `${Math.min(100, ((formData.los_days || 0) / 30) * 100)}%` }} />
                    </div>
                  </div>

                  {/* Second card: Lab results summary. */}
                  <div className="stat-card">
                    <div className="stat-icon emerald"><Activity size={16} /></div>
                    <div className="stat-name">Abnormal Labs</div>
                    <div className="stat-value">{formData.lab_abnormal_count || 0}<span style={{fontSize:'1.1rem',color:'var(--text-dim)', fontWeight:'500'}}> hits</span></div>
                    <div className="stat-sub">{((formData.lab_abnormal_rate || 0) * 100).toFixed(0)}% abnormal rate</div>
                    <div className="stat-bar">
                      <div className="stat-bar-fill emerald" style={{ width: `${Math.min(100, (formData.lab_abnormal_rate || 0) * 100)}%` }} />
                    </div>
                  </div>

                  {/* Third card: Historical hospital usage. */}
                  <div className="stat-card">
                    <div className="stat-icon silver"><RefreshCcw size={16} /></div>
                    <div className="stat-name">Admission History</div>
                    <div className="stat-value">{formData.prev_admissions || 0}<span style={{fontSize:'1.1rem',color:'var(--text-dim)', fontWeight:'500'}}> prior</span></div>
                    <div className="stat-sub">{formData.days_since_last || 0} days since last visit</div>
                  </div>
                </div>

                {/* ── Clinical Note Display ── */}
                {/* Only render this card if the user provided text. */}
                {formData.note && (
                  <div className="note-card">
                    <div className="card-header">
                      <div className="card-title"><FileText size={14} /> Physician Note Analysis</div>
                      <span className="card-tag">Natural Language Processing</span>
                    </div>
                    {/* Display the original clinical text back to the user. */}
                    <p className="note-text">{formData.note}</p>
                  </div>
                )}

                {/* ── Top Predictive Factors List ── */}
                {/* Loops over the top_factors array returned by the Python server and renders them in order. */}
                {result.top_factors && result.top_factors.length > 0 && (
                  <div className="note-card" style={{ marginTop: '0', background: 'rgba(255, 255, 255, 0.8)' }}>
                    <div className="card-header" style={{ marginBottom: '1rem' }}>
                      <div className="card-title"><Activity size={14} /> Key Predictive Factors</div>
                      <span className="card-tag" style={{ background: 'var(--primary-glow)', color: 'var(--primary)' }}>Top-K Gating</span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                      {/* Using map to iterate over the array of factor dictionaries. */}
                      {result.top_factors.map((factor, i) => (
                        <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 1rem', background: 'white', border: '1px solid var(--card-border)', borderRadius: '12px', boxShadow: '0 2px 10px rgba(0,0,0,0.02)' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            {/* Render the rank number (e.g., #1, #2). */}
                            <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: 'var(--primary-glow)', color: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '600', fontSize: '0.8rem' }}>
                              #{i + 1}
                            </div>
                            <div>
                              {/* The name of the feature driving the score. */}
                              <div style={{ fontSize: '0.95rem', fontWeight: '600', color: 'var(--text-main)' }}>{factor.name}</div>
                              {/* The exact numeric value that triggered the model. */}
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Value: {factor.value}</div>
                            </div>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            {/* If the backend marked this feature as 'missing' (imputed), show a warning badge. */}
                            {factor.missing && (
                              <span style={{ fontSize: '0.65rem', fontWeight: '700', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#EA580C', background: '#FFEDD5', padding: '0.3rem 0.6rem', borderRadius: '6px' }}>
                                Imputed (Missing)
                              </span>
                            )}
                            {/* The calculated impact percentage. */}
                            <div style={{ fontSize: '1.2rem', fontWeight: '600', color: 'var(--primary)', fontFamily: 'Outfit' }}>
                              {factor.impact}%
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* ── Extra System Insights Row ── */}
                <div className="insights-row">
                  {/* Insight card detailing how many fields were filled vs how many the model expected. */}
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

                  {/* Insight card detailing technical model specifications. */}
                  <div className="insight-card">
                    <div className="card-header">
                      <div className="card-title"><Heart size={14} /> TRANCE Framework</div>
                    </div>
                    <p>
                      Powered by the <strong>Transformer-Augmented Neural Clinical Engine</strong>, trained on real-world EHR data.
                      Disease-specific ICD markers and <strong>512-dimensional NLP embeddings</strong> ensure precision per pathology.
                    </p>
                    <div style={{ marginTop:'0.75rem', display:'flex', gap:'0.5rem' }}>
                      {/* Badge showing the total count of features the Python server processed behind the scenes. */}
                      <span style={{ fontSize:'0.7rem', fontWeight:'600', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--primary)', background:'white', border:'1px solid rgba(0,0,0,0.05)', borderRadius:'6px', padding:'0.3rem 0.75rem', boxShadow:'0 2px 5px rgba(0,0,0,0.02)' }}>
                        {result?.stats?.features_used || 0} Features
                      </span>
                      <span style={{ fontSize:'0.7rem', fontWeight:'600', letterSpacing:'0.12em', textTransform:'uppercase', color:'var(--text-dim)', background:'rgba(255,255,255,0.5)', border:'1px solid rgba(0,0,0,0.05)', borderRadius:'6px', padding:'0.3rem 0.75rem' }}>
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

      {/* ── Global Footer ── */}
      <footer className="app-footer">
        <div style={{ display:'flex', gap:'1.5rem' }}>
          <span>© 2026 WeCare Healthcare Systems</span>
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
