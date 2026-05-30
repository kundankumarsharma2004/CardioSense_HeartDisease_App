/* ─── CardioSense Frontend ──────────────────────────── */

const MODEL_COLORS = {
  'Logistic Regression': '#2563EB',
  'Random Forest':       '#10B981',
  'SVM':                 '#F59E0B',
  'KNN':                 '#EF4444',
};

const FEATURE_LABELS = {
  age: 'Age', sex: 'Sex', cp: 'Chest Pain', trestbps: 'Resting BP',
  chol: 'Cholesterol', fbs: 'Fasting BS', restecg: 'Rest ECG',
  thalach: 'Max Heart Rate', exang: 'Exercise Angina',
  oldpeak: 'ST Depression', slope: 'ST Slope', ca: 'Major Vessels',
  thal: 'Thalassemia'
};

let statsData = null;      // cached from /api/stats
let chartsLoaded = {};     // track which charts are rendered

// ─── TAB SWITCHING ───────────────────────────────────
const TAB_TITLES = {
  predict: 'Patient Prediction',
  dashboard: 'Model Dashboard',
  analytics: 'Analytics',
  about: 'About',
};

function switchTab(tab, el) {
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
  el.classList.add('active');
  document.getElementById('topbar-title').textContent = TAB_TITLES[tab];

  if (tab === 'dashboard' && !chartsLoaded.dashboard) loadDashboard();
  if (tab === 'analytics' && !chartsLoaded.analytics) loadAnalytics();
}

// ─── SIDEBAR TOGGLE (mobile) ─────────────────────────
function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('open');
}

// ─── FETCH STATS (shared) ────────────────────────────
async function fetchStats() {
  if (statsData) return statsData;
  const res = await fetch('/api/stats');
  statsData = await res.json();
  return statsData;
}

// ─── PREDICTION ──────────────────────────────────────
async function runPrediction() {
  const btn = document.querySelector('.btn-primary');
  btn.classList.add('loading');
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation:spin .7s linear infinite"><path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4"/></svg> Analysing…`;

  const fields = ['age','sex','cp','trestbps','chol','fbs','restecg',
                  'thalach','exang','oldpeak','slope','ca','thal'];
  const payload = {};
  for (const f of fields) {
    const el = document.getElementById('f-' + f);
    payload[f] = parseFloat(el.value);
  }

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    renderResults(data);
  } catch (e) {
    alert('Prediction failed: ' + e.message);
  } finally {
    btn.classList.remove('loading');
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> Run Prediction`;
  }
}

function renderResults(data) {
  document.getElementById('results-placeholder').style.display = 'none';
  const out = document.getElementById('results-output');
  out.style.display = 'block';

  // Consensus
  const cc   = document.getElementById('consensus-card');
  const risk = data.consensus;
  document.getElementById('consensus-risk').textContent = risk === 'HIGH' ? '⚠ High Risk' : '✓ Low Risk';
  document.getElementById('consensus-risk').className = 'consensus-risk ' + risk.toLowerCase();
  document.getElementById('consensus-prob').textContent =
    `Average Probability: ${(data.avg_probability * 100).toFixed(1)}%`;
  document.getElementById('consensus-vote').textContent =
    `${data.high_votes} of ${data.total_models} models predict Heart Disease`;

  const pct = data.avg_probability * 100;
  document.getElementById('gauge-fill').style.width = pct + '%';
  document.getElementById('gauge-thumb').style.left  = pct + '%';

  // Model cards
  const grid = document.getElementById('model-cards-grid');
  grid.innerHTML = '';
  let delay = 0;
  for (const [name, info] of Object.entries(data.predictions)) {
    const col   = MODEL_COLORS[name] || '#64748B';
    const pct   = (info.probability * 100).toFixed(1);
    const card  = document.createElement('div');
    card.className = 'model-card';
    card.style.cssText = `--model-color:${col}; animation-delay:${delay}ms`;
    card.innerHTML = `
      <div class="model-card-name">${name}</div>
      <div class="model-prob">${pct}<span style="font-size:.9rem;color:var(--text-3)">%</span></div>
      <span class="model-risk-badge ${info.risk === 'HIGH' ? 'risk-high' : 'risk-low'}">
        ${info.risk === 'HIGH' ? '▲ HIGH RISK' : '✓ LOW RISK'}
      </span>
      <div class="model-prob-bar">
        <div class="model-prob-fill" style="width:${pct}%;background:${col}"></div>
      </div>`;
    grid.appendChild(card);
    delay += 80;
  }
}

function resetForm() {
  document.getElementById('results-placeholder').style.display = 'flex';
  document.getElementById('results-output').style.display = 'none';

  const defaults = {
    age:54, sex:1, cp:2, trestbps:130, chol:250, fbs:0,
    restecg:1, thalach:155, exang:0, oldpeak:1.5, slope:1, ca:0, thal:2
  };
  for (const [f, v] of Object.entries(defaults)) {
    document.getElementById('f-' + f).value = v;
  }
}

// ─── DASHBOARD ───────────────────────────────────────
async function loadDashboard() {
  chartsLoaded.dashboard = true;
  const data = await fetchStats();

  // Stat pills
  const ds = data.dataset;
  const perf = data.performance;
  const bestAcc = Math.max(...Object.values(perf).map(m => m.accuracy));
  const bestAuc = Math.max(...Object.values(perf).map(m => m.auc));

  setText('stat-total',     ds.total.toLocaleString());
  setText('stat-disease',   ds.disease.toLocaleString());
  setText('stat-nodisease', ds.no_disease.toLocaleString());
  setText('stat-best-acc',  (bestAcc * 100).toFixed(1) + '%');
  setText('stat-best-auc',  bestAuc.toFixed(4));

  document.querySelectorAll('.stat-pill').forEach(p => p.classList.remove('loading-pulse'));

  // Performance table
  const tbody = document.getElementById('perf-tbody');
  tbody.innerHTML = '';
  let bestAccModel = Object.entries(perf).sort((a,b) => b[1].accuracy - a[1].accuracy)[0][0];
  for (const [name, m] of Object.entries(perf)) {
    const col = MODEL_COLORS[name];
    const tr  = document.createElement('tr');
    if (name === bestAccModel) tr.className = 'best-row';
    tr.innerHTML = `
      <td><span class="model-dot" style="background:${col}"></span>${name}${name===bestAccModel ? ' 🏆' : ''}</td>
      <td>${(m.accuracy*100).toFixed(2)}%</td>
      <td>${m.auc.toFixed(4)}</td>
      <td>${(m.cv_score*100).toFixed(2)}%</td>
      <td>${m.precision.toFixed(4)}</td>
      <td>${m.recall.toFixed(4)}</td>
      <td>${m.f1.toFixed(4)}</td>`;
    tbody.appendChild(tr);
  }

  // Charts
  loadChart('/api/chart/accuracy_bar', 'chart-accuracy');
  loadChart('/api/chart/roc',          'chart-roc');
  loadChart('/api/chart/confusion',    'chart-confusion');
}

// ─── ANALYTICS ───────────────────────────────────────
async function loadAnalytics() {
  chartsLoaded.analytics = true;
  const data = await fetchStats();

  // Feature importance chart
  loadChart('/api/chart/feature_importance', 'chart-feature');

  // Feature list
  const fi  = data.feature_importance;
  const sorted = Object.entries(fi).sort((a,b) => b[1]-a[1]);
  const maxVal = sorted[0][1];
  const list = document.getElementById('feature-list');
  list.innerHTML = '<div class="feature-list-inner">' +
    sorted.map(([key, val], i) => `
      <div class="feature-row">
        <span class="feature-rank">${i+1}</span>
        <span class="feature-name">${FEATURE_LABELS[key] || key}</span>
        <div class="feature-bar-wrap">
          <div class="feature-bar-fill" style="width:${(val/maxVal*100).toFixed(1)}%"></div>
        </div>
        <span class="feature-score">${val.toFixed(3)}</span>
      </div>`
    ).join('') + '</div>';

  // Prob dist chart
  loadChart('/api/chart/prob_dist', 'chart-prob-dist');
}

// ─── CHART LOADER ─────────────────────────────────────
async function loadChart(url, containerId) {
  const container = document.getElementById(containerId);
  container.innerHTML = '<div class="chart-loader">Loading chart…</div>';
  try {
    const res  = await fetch(url);
    const data = await res.json();
    const img  = document.createElement('img');
    img.src = 'data:image/png;base64,' + data.image;
    img.alt = 'chart';
    img.style.opacity = 0;
    img.style.transition = 'opacity .4s';
    img.onload = () => { img.style.opacity = 1; };
    container.innerHTML = '';
    container.appendChild(img);
  } catch (e) {
    container.innerHTML = `<div class="chart-loader" style="color:#EF4444">Failed to load chart.</div>`;
  }
}

// ─── UTILS ───────────────────────────────────────────
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Spin animation for loading button
const style = document.createElement('style');
style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
document.head.appendChild(style);
