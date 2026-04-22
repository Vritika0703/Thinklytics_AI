/**
 * Decision Assistant AI — Main Controller v2.0
 * 
 * New in this version:
 *   - Reasoning Summary panel
 *   - Bad input detection & clarity warning
 *   - Decision history (localStorage + /history API)
 *   - Confidence score visualized as a live counter
 */

const CONFIG = {
    API_URL: 'http://localhost:8001',
    STAGES: [
        { text: "Parsing your situation...",               progress: 18 },
        { text: "Evaluating strategic implications...",    progress: 38 },
        { text: "Calculating risk vectors...",             progress: 57 },
        { text: "Synthesizing perspectives...",            progress: 76 },
        { text: "Assembling final analysis...",            progress: 95 },
    ]
};

let currentAnalysis = null;

// ── Init ──────────────────────────────────────────────────────────────────────
let el = {};  // populated in DOMContentLoaded

document.addEventListener('DOMContentLoaded', () => {
    // Build el refs after DOM is ready (avoids null on fast loads)
    el = {
        header:           document.getElementById('header-section'),
        inputSection:     document.getElementById('input-section'),
        situationInput:   document.getElementById('situation-input'),
        analyzeBtn:       document.getElementById('analyze-btn'),
        clarityWarning:   document.getElementById('clarity-warning'),
        clarityMessage:   document.getElementById('clarity-message'),
        historyToggle:    document.getElementById('history-toggle-btn'),
        historySection:   document.getElementById('history-section'),
        historyClose:     document.getElementById('history-close-btn'),
        clearHistoryBtn:  document.getElementById('clear-history-btn'),
        historyList:      document.getElementById('history-list'),
        loadingSection:   document.getElementById('loading-section'),
        thinkingStep:     document.getElementById('thinking-step'),
        progressBar:      document.getElementById('progress-bar'),
        resultsDashboard: document.getElementById('results-dashboard'),
        confScore:        document.getElementById('conf-score'),
        logicBar:         document.getElementById('logic-bar'),
        logicVal:         document.getElementById('logic-val'),
        emotionBar:       document.getElementById('emotion-bar'),
        emotionVal:       document.getElementById('emotion-val'),
        reasoningSummary: document.getElementById('reasoning-summary'),
        mlTags:           document.getElementById('ml-tags'),
        recommendText:    document.getElementById('recommendation-text'),
        prosList:         document.getElementById('pros-list'),
        consList:         document.getElementById('cons-list'),
        risksList:        document.getElementById('risks-list'),
        perspectiveContent: document.getElementById('perspective-content'),
        resetBtn:         document.getElementById('reset-btn'),
    };

    lucide.createIcons();
    bindEvents();
    showSection('input');
    loadHistory();
    checkAIStatus();
});

// ── DOM helper ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Events ────────────────────────────────────────────────────────────────
function bindEvents() {
    el.analyzeBtn.addEventListener('click', handleAnalysis);
    el.resetBtn.addEventListener('click', () => showSection('input'));
    el.historyToggle.addEventListener('click', toggleHistory);
    el.historyClose.addEventListener('click', toggleHistory);
    el.clearHistoryBtn.addEventListener('click', clearHistory);

    // Ctrl+Enter or Cmd+Enter submits
    el.situationInput.addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') handleAnalysis();
    });

    document.querySelectorAll('.perspective-tab').forEach(tab => {
        tab.addEventListener('click', e => {
            document.querySelectorAll('.perspective-tab').forEach(t => t.classList.remove('active'));
            e.currentTarget.classList.add('active');
            updatePerspective(e.currentTarget.dataset.perspective);
        });
    });

    // Clear warning on new input
    el.situationInput.addEventListener('input', () => {
        el.clarityWarning.classList.add('hidden');
    });
}

// ── Analysis Flow ────────────────────────────────────────────────────────────────
let _thinkingActive = false;

async function handleAnalysis() {
    const situation = el.situationInput.value.trim();
    if (!situation || situation.length < 3) {
        showClarityWarning('Please describe your situation before analyzing.');
        return;
    }

    // Disable button to prevent double-submit
    el.analyzeBtn.disabled = true;
    el.analyzeBtn.innerHTML = '<i class="spin-icon" data-lucide="loader-circle"></i> <span>Analyzing...</span>';
    lucide.createIcons();

    showSection('loading');
    _thinkingActive = true;

    // Start looping animation in the background
    simulateThinking();

    // Await ONLY the API call
    const analysis = await callAPI(situation);

    // Stop animation loop
    _thinkingActive = false;

    // Restore button
    el.analyzeBtn.disabled = false;
    el.analyzeBtn.innerHTML = '<span>Analyze Situation</span> <i data-lucide="arrow-right"></i>';
    lucide.createIcons();

    if (!analysis) {
        showSection('input');
        return;
    }

    if (!analysis.is_clear) {
        showSection('input');
        showClarityWarning(analysis.clarity_message || 'Please describe your situation in more detail.');
        return;
    }

    currentAnalysis = analysis;
    renderResults(analysis);
    showSection('results');
    loadHistory();
}

async function callAPI(situation) {
    try {
        const res = await fetch(`${CONFIG.API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ situation })
        });

        const data = await res.json();

        if (!res.ok) {
            // Server returned an error — show clarity warning instead of alert
            const msg = data?.detail || 'Something went wrong. Please try again.';
            showClarityWarning(msg);
            return null;
        }

        return data;
    } catch (err) {
        console.error('API call failed:', err);
        showClarityWarning('Cannot reach the backend server. Make sure it is running on port 8001.');
        return null;
    }
}

async function simulateThinking() {
    const extended = [
        { text: 'Running deeper analysis...', progress: 30 },
        { text: 'Cross-referencing risk patterns...', progress: 55 },
        { text: 'Finalizing recommendation...', progress: 78 },
        { text: 'Almost there — polishing insights...', progress: 90 },
    ];
    const startTime = Date.now();
    let loop = 0;

    while (_thinkingActive) {
        const elapsed = Date.now() - startTime;
        const stages = elapsed > 8000 ? extended : CONFIG.STAGES;

        for (const stage of stages) {
            if (!_thinkingActive) break;
            el.thinkingStep.textContent = stage.text;
            el.progressBar.style.width = `${stage.progress}%`;
            await pause(800);
        }
        loop++;
    }

    // Show complete state
    el.progressBar.style.width = '100%';
}

// ── Rendering ─────────────────────────────────────────────────────────────────
function renderResults(data) {
    // ── Confidence counter ──
    animateCounter(el.confScore, 0, Math.round(data.confidence_score * 100), '%', 1200);

    // ── Bars and labels ──
    const logicPct   = Math.round(data.logical_score * 100);
    const emotionPct = Math.round(data.emotional_score * 100);

    // Reset to 0 first so animation always replays on new result
    el.logicBar.style.width   = '0%';
    el.emotionBar.style.width = '0%';
    el.logicVal.textContent   = '—';
    el.emotionVal.textContent = '—';

    setTimeout(() => { el.logicBar.style.width   = `${logicPct}%`; }, 100);
    setTimeout(() => { el.emotionBar.style.width = `${emotionPct}%`; }, 300);

    // Labels reveal AFTER bar animation completes (1.2s transition + buffer)
    setTimeout(() => {
        el.logicVal.textContent   = logicPct >= 70 ? 'High' : logicPct >= 40 ? 'Moderate' : 'Low';
        el.emotionVal.textContent = emotionPct >= 70 ? 'High' : emotionPct >= 40 ? 'Balanced' : 'Low';
    }, 1500);

    // ── Text & ML Tags ──
    el.reasoningSummary.textContent = data.reasoning_summary;
    el.recommendText.textContent = data.structured_recommendation;

    let tagHTML = '<span class="tag tag-blue">AI Reasoning</span>';
    if (data.ml_category) tagHTML += `<span class="tag tag-purple">ML Cat: ${data.ml_category}</span>`;
    if (data.ml_risk)     tagHTML += `<span class="tag tag-purple">ML Risk: ${data.ml_risk}</span>`;
    if (data.ml_sentiment)tagHTML += `<span class="tag tag-purple">ML Sent: ${data.ml_sentiment}</span>`;
    if (el.mlTags) el.mlTags.innerHTML = tagHTML;

    // ── Pros ──
    el.prosList.innerHTML = data.pros.map(p => `
        <li class="item-row">
            <i data-lucide="check" style="color:#4ade80"></i>
            <span>${p.text}</span>
        </li>`).join('');

    // ── Cons ──
    el.consList.innerHTML = data.cons.map(c => `
        <li class="item-row">
            <i data-lucide="x" style="color:#f87171"></i>
            <span>${c.text}</span>
        </li>`).join('');

    // ── Risks ──
    el.risksList.innerHTML = data.risks.map(r => `
        <li class="item-row" style="flex-direction:column;gap:6px;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
                <span>${r.text}</span>
                <span class="severity-badge sev-${r.severity}">${r.severity}</span>
            </div>
        </li>`).join('');

    // ── Default perspective ──
    updatePerspective('practical');
    document.querySelectorAll('.perspective-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.perspective === 'practical');
    });

    lucide.createIcons();
}

function updatePerspective(type) {
    if (!currentAnalysis) return;
    const map = {
        practical:  currentAnalysis.practical_perspective,
        optimistic: currentAnalysis.optimistic_perspective,
        worst_case: currentAnalysis.worst_case_perspective,
    };
    const text = map[type] || 'No perspective available.';

    // Fade transition
    el.perspectiveContent.style.opacity = '0';
    setTimeout(() => {
        el.perspectiveContent.textContent = text;
        el.perspectiveContent.style.opacity = '1';
    }, 180);
    el.perspectiveContent.style.transition = 'opacity 0.18s ease';
}

// ── History ───────────────────────────────────────────────────────────────────
let _historyData = [];

async function loadHistory() {
    try {
        const res = await fetch(`${CONFIG.API_URL}/history`);
        if (!res.ok) return;
        const data = await res.json();
        _historyData = data.decisions || [];
        renderHistory(_historyData);
    } catch (_) {
        // Server might not be running yet — render from nothing
        renderHistory([]);
    }
}

function renderHistory(decisions) {
    if (!decisions.length) {
        el.historyList.innerHTML = '<li class="history-empty">No decisions analyzed yet.</li>';
        return;
    }

    el.historyList.innerHTML = decisions.map((d, index) => `
        <li class="history-item" onclick="openHistoryItem(${index})">
            <div class="history-item-date">${formatDate(d.timestamp)}</div>
            <div class="history-item-situation">${truncate(d.situation, 100)}</div>
            <div class="history-item-score">
                <span class="score-badge">${Math.round(d.confidence_score * 100)}% confidence</span>
            </div>
            <div class="history-item-rec">${truncate(d.recommendation, 120)}</div>
        </li>
    `).join('');
}

window.openHistoryItem = function(index) {
    const analysis = _historyData[index];
    if (!analysis) return;
    
    // Set UI State and load it
    currentAnalysis = analysis;
    el.historySection.classList.add('hidden');
    renderResults(analysis);
    showSection('results');
};

async function clearHistory() {
    try {
        await fetch(`${CONFIG.API_URL}/history`, { method: 'DELETE' });
        renderHistory([]);
    } catch (_) {
        renderHistory([]);
    }
}

function toggleHistory() {
    el.historySection.classList.toggle('hidden');
    lucide.createIcons();
}

// ── Section control ───────────────────────────────────────────────────────────
function showSection(name) {
    el.inputSection.classList.add('hidden');
    el.loadingSection.classList.add('hidden');
    el.resultsDashboard.classList.add('hidden');
    el.header.style.display = '';

    if (name === 'input') {
        el.inputSection.classList.remove('hidden');
        gsap.from(el.inputSection, { opacity: 0, y: 20, duration: 0.6, ease: 'power2.out' });
        el.progressBar.style.width = '0%';
    } else if (name === 'loading') {
        el.header.style.display = 'none';
        el.historySection.classList.add('hidden');
        el.loadingSection.classList.remove('hidden');
        gsap.from(el.loadingSection, { opacity: 0, scale: 0.96, duration: 0.5 });
    } else if (name === 'results') {
        el.header.style.display = 'none';
        el.resultsDashboard.classList.remove('hidden');
        gsap.from('.results-dashboard > *', {
            opacity: 0, y: 30, stagger: 0.1, duration: 0.8, ease: 'expo.out'
        });
        lucide.createIcons();
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showClarityWarning(msg) {
    el.clarityMessage.textContent = msg;
    el.clarityWarning.classList.remove('hidden');
    gsap.from(el.clarityWarning, { opacity: 0, y: -8, duration: 0.4 });
}

function animateCounter(el, from, to, suffix, duration) {
    const start = performance.now();
    const update = (now) => {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(from + (to - from) * eased) + suffix;
        if (t < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
}

function pause(ms) { return new Promise(r => setTimeout(r, ms)); }

async function checkAIStatus() {
    const banner   = $('mode-banner');
    const bannerText = $('mode-text');
    const keyLink  = $('get-key-link');

    try {
        const res = await fetch(`${CONFIG.API_URL}/status`);
        if (!res.ok) throw new Error('no response');
        const data = await res.json();

        banner.classList.remove('hidden', 'demo', 'live');

        if (data.live_mode) {
            banner.classList.add('live');
            bannerText.textContent = `✦ Live AI — powered by ${data.provider.charAt(0).toUpperCase() + data.provider.slice(1)}`;
            keyLink.classList.add('hidden');
        } else {
            banner.classList.add('demo');
            bannerText.textContent = '⚠ Demo mode — results are illustrative, not AI-generated.';
            keyLink.classList.remove('hidden');
        }
        lucide.createIcons();
    } catch (_) {
        // Backend not running — show nothing
    }
}


function formatDate(iso) {
    if (!iso) return '';
    // Ensure "Z" UTC terminator so browsers interpret as local correctly
    if (!iso.endsWith('Z') && !iso.includes('+')) iso += 'Z';
    const d = new Date(iso);
    return d.toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function truncate(str, n) {
    if (!str) return '';
    return str.length > n ? str.slice(0, n) + '…' : str;
}
