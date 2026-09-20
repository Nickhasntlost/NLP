import React, { useState, useEffect } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const SAMPLE_PRESETS = [
  { label: 'Elongations', text: 'bhaaaai kyaaa scene hai kal ka?' },
  { label: 'Slang Contractions', text: 'Aaj ka din bht zyada thk gya hu bhai' },
  { label: 'Polite Multi-clause', text: 'Sir aap bahut achha padhate ho, mujhe samajh aa gaya.' },
  { label: 'Social & Copula', text: 'koi nhi bolega scripted h ye video' },
  { label: 'Loanwords & Numbers', text: 'Maine 636 tiktok videos khud report kiye the' },
  { label: 'Pure Devanagari', text: 'भाई क्या कर रहे हो आजकल?' },
];

export default function App() {
  const [inputText, setInputText] = useState('bhaaaai kyaaa scene hai kal ka?');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);
  const [telemetryOpen, setTelemetryOpen] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [useNeuralGrammar, setUseNeuralGrammar] = useState(false);

  // Check API health on load
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (res.ok) {
          setApiOnline(true);
        } else {
          setApiOnline(false);
        }
      } catch (err) {
        setApiOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleTranslate = async (textToTranslate = inputText) => {
    const text = textToTranslate.trim();
    if (!text) {
      setError('Please enter a Hinglish or Hindi sentence to translate.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/translate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          use_neural_grammar: useNeuralGrammar,
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResult(data);
      setTelemetryOpen(true);
    } catch (err) {
      setError(err.message || 'Failed to connect to translation backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!result?.final_translation) return;
    navigator.clipboard.writeText(result.final_translation);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClear = () => {
    setInputText('');
    setResult(null);
    setError(null);
  };

  const selectPreset = (text) => {
    setInputText(text);
    handleTranslate(text);
  };

  return (
    <div className="app-container">
      <div className="ambient-glow" />

      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon-box">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m5 8 6 6" />
              <path d="m4 14 6-6 2-3" />
              <path d="M2 5h12" />
              <path d="M7 2h1" />
              <path d="m22 22-5-10-5 10" />
              <path d="M14 18h6" />
            </svg>
          </div>
          <div>
            <h1 className="brand-title">Beyond Words</h1>
            <p className="brand-subtitle">Context-Aware Hinglish to English Neural Translation</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="status-badge" title={apiOnline ? 'FastAPI backend connected' : 'Connecting to http://localhost:8000'}>
            <span className={`status-dot ${apiOnline ? '' : 'offline'}`} />
            <span>{apiOnline ? 'Engine Online' : 'Engine Offline'}</span>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            <input
              type="checkbox"
              checked={useNeuralGrammar}
              onChange={(e) => setUseNeuralGrammar(e.target.checked)}
              style={{ accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
            />
            <span>Deep Neural Polish</span>
          </label>
        </div>
      </header>

      {/* Presets Row */}
      <div className="presets-section">
        <div className="presets-label">Try Real-World Test Samples:</div>
        <div className="preset-pills-list">
          {SAMPLE_PRESETS.map((preset, idx) => (
            <button
              key={idx}
              className="preset-pill"
              onClick={() => selectPreset(preset.text)}
              disabled={loading}
            >
              <strong style={{ color: '#c7d2fe', marginRight: '4px' }}>{preset.label}:</strong> "{preset.text}"
            </button>
          ))}
        </div>
      </div>

      {/* Main Dual Workspace */}
      <main className="workspace-grid">
        {/* Left: Input Card */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 7V4h16v3M9 20h6M12 4v16"/></svg>
              <span>Hinglish / Hindi Source</span>
            </div>
            <span className="lang-tag">Roman + Devanagari</span>
          </div>

          <textarea
            className="text-input-area"
            placeholder="Type or paste Romanized Hinglish or Devanagari Hindi (e.g., 'Bhai kya kar raha hai aajkal?')..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                handleTranslate();
              }
            }}
          />

          <div className="card-footer">
            <div className="char-counter">{inputText.length} characters</div>
            <div style={{ display: 'flex', gap: '0.6rem' }}>
              {inputText && (
                <button className="action-btn" onClick={handleClear} disabled={loading} title="Clear text">
                  Clear
                </button>
              )}
              <button
                className="action-btn primary-btn"
                onClick={() => handleTranslate()}
                disabled={loading || !inputText.trim()}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    <span>Translating...</span>
                  </>
                ) : (
                  <>
                    <span>Translate</span>
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right: Output Card */}
        <div className="glass-card">
          <div className="card-header">
            <div className="card-title">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m5 12 5 5L20 7"/></svg>
              <span>English Translation</span>
            </div>
            <span className="lang-tag" style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#34d399', borderColor: 'rgba(16, 185, 129, 0.25)' }}>
              Fluent English
            </span>
          </div>

          <div className={`text-output-area ${!result?.final_translation ? 'empty' : ''}`}>
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                <span className="spinner" /> Generating translation via IndicTrans2 1B...
              </span>
            ) : result?.final_translation ? (
              result.final_translation
            ) : (
              'Translation will appear here in natural English.'
            )}
          </div>

          <div className="card-footer">
            <div className="char-counter">
              {result?.timing_ms?.total_ms ? (
                <span style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                  ⚡ {result.timing_ms.total_ms.toFixed(1)} ms
                </span>
              ) : (
                '5-Stage Pipeline'
              )}
            </div>

            {result?.final_translation && (
              <button className="action-btn" onClick={handleCopy} title="Copy to clipboard">
                {copied ? (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" strokeWidth="2.5"><path d="M20 6 9 17l-5-5"/></svg>
                    <span style={{ color: '#34d399' }}>Copied!</span>
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>
                    <span>Copy</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      </main>

      {/* Error Banner */}
      {error && (
        <div style={{ background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.35)', color: '#fca5a5', padding: '0.85rem 1.25rem', borderRadius: 'var(--radius-md)', marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
          <span>{error}</span>
        </div>
      )}

      {/* Pipeline Telemetry Inspector Accordion */}
      {result && (
        <section className="telemetry-card">
          <div
            className={`telemetry-header ${telemetryOpen ? 'expanded' : ''}`}
            onClick={() => setTelemetryOpen(!telemetryOpen)}
          >
            <div className="telemetry-title-group">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-purple)" strokeWidth="2"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
              <span style={{ fontWeight: 600, fontFamily: 'var(--font-display)', fontSize: '1.05rem' }}>
                Pipeline Telemetry & Stage Transformations
              </span>
              <span className="latency-badge">
                Total: {result.timing_ms?.total_ms ? `${result.timing_ms.total_ms.toFixed(1)} ms` : 'Complete'}
              </span>
            </div>
            <svg
              width="18"
              height="18"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              style={{ transform: telemetryOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s ease' }}
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </div>

          {telemetryOpen && (
            <div className="telemetry-body">
              {/* Step 1: Preprocessing */}
              <div className="telemetry-step-box">
                <div className="step-label">
                  <span>1. Preprocessing (Phase 2)</span>
                  <span className="step-time">{result.timing_ms?.preprocessing_ms ?? 0} ms</span>
                </div>
                <div className="step-content">
                  Tokens ({result.preprocessed_tokens?.length || 0}):<br />
                  {JSON.stringify(result.preprocessed_tokens)}
                </div>
              </div>

              {/* Step 2: Language Identification */}
              <div className="telemetry-step-box">
                <div className="step-label">
                  <span>2. Token LID (Phase 3)</span>
                  <span className="step-time">{result.timing_ms?.lid_ms ?? 0} ms</span>
                </div>
                <div className="tag-cloud">
                  {result.lid_tags?.map(([token, tag], idx) => (
                    <span key={idx} className={`token-tag ${tag}`} title={`Script: ${tag}`}>
                      {token} <small style={{ opacity: 0.7 }}>[{tag}]</small>
                    </span>
                  ))}
                </div>
              </div>

              {/* Step 3: Normalization */}
              <div className="telemetry-step-box">
                <div className="step-label">
                  <span>3. Normalization (Phase 5)</span>
                  <span className="step-time">{result.timing_ms?.normalization_ms ?? 0} ms</span>
                </div>
                <div className="step-content">
                  Normalized: "{result.normalized_hinglish}"
                </div>
              </div>

              {/* Step 4: Transliteration & Model 3 */}
              <div className="telemetry-step-box">
                <div className="step-label">
                  <span>4. IndicTrans2 1B (Phase 4)</span>
                  <span className="step-time">{result.timing_ms?.translation_ms ?? 0} ms</span>
                </div>
                <div className="step-content">
                  Devanagari Input: {result.devanagari_input}<br />
                  Raw Output: "{result.raw_translation}"
                </div>
              </div>

              {/* Step 5: Grammar Correction */}
              <div className="telemetry-step-box">
                <div className="step-label">
                  <span>5. Grammar Polish (Phase 6)</span>
                  <span className="step-time">{result.timing_ms?.grammar_ms ?? 0} ms</span>
                </div>
                <div className="step-content">
                  Final: "{result.final_translation}"
                </div>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Footer */}
      <footer className="app-footer">
        Beyond Words &bull; ai4bharat/indictrans2-indic-en-1B baseline &bull; Fast Rule & Neural Post-Processing &bull; End-to-End Hinglish MT
      </footer>
    </div>
  );
}
