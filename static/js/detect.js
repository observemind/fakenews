/**
 * detect.js — Detection page interactions
 * Handles API call, result animation, progress bar, examples
 */

'use strict';

// ── Character counter ────────────────────────────────────────────────────────
const textarea = document.getElementById('newsInput');
const charCount = document.getElementById('charCount');

if (textarea && charCount) {
  textarea.addEventListener('input', () => {
    charCount.textContent = textarea.value.length;
    // Pulse on update
    charCount.style.color = textarea.value.length > 4500 ? 'var(--neon-red)' : 'var(--text-muted)';
  });
}

// ── Load example text ────────────────────────────────────────────────────────
function loadExample(card) {
  const text = card.querySelector('p').textContent;
  textarea.value = text;
  charCount.textContent = text.length;
  textarea.focus();
  textarea.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // Highlight textarea briefly
  textarea.style.borderColor = 'var(--neon-purple)';
  textarea.style.boxShadow = '0 0 0 3px rgba(139,92,246,0.15), 0 0 20px rgba(139,92,246,0.35)';
  setTimeout(() => {
    textarea.style.borderColor = '';
    textarea.style.boxShadow = '';
  }, 1200);
}

// ── Clear input ───────────────────────────────────────────────────────────────
function clearInput() {
  textarea.value = '';
  charCount.textContent = '0';
  textarea.focus();
  hideResult();
}

function hideResult() {
  const resultPanel = document.getElementById('resultPanel');
  if (resultPanel) {
    resultPanel.style.opacity = '0';
    resultPanel.style.transform = 'translateY(20px)';
    setTimeout(() => { resultPanel.style.display = 'none'; }, 300);
  }
}

// ── Animated progress bar ─────────────────────────────────────────────────────
function showProgress() {
  const wrap = document.getElementById('progressWrap');
  const bar = document.getElementById('progressBar');
  if (!wrap || !bar) return;

  wrap.style.display = 'block';
  bar.style.width = '0%';

  const steps = [
    { width: '15%', label: 'Tokenising text...', delay: 100 },
    { width: '35%', label: 'Removing stopwords...', delay: 400 },
    { width: '55%', label: 'Stemming tokens...', delay: 700 },
    { width: '75%', label: 'Computing TF-IDF vectors...', delay: 1000 },
    { width: '90%', label: 'Running classifier...', delay: 1300 },
    { width: '100%', label: 'Finalising result...', delay: 1600 },
  ];

  const label = wrap.querySelector('.progress-label');
  steps.forEach(({ width, label: text, delay }) => {
    setTimeout(() => {
      bar.style.transition = 'width 0.4s cubic-bezier(0.22, 1, 0.36, 1)';
      bar.style.width = width;
      if (label) label.textContent = text;
    }, delay);
  });
}

function hideProgress() {
  const wrap = document.getElementById('progressWrap');
  if (wrap) {
    setTimeout(() => { wrap.style.display = 'none'; }, 200);
  }
}

// ── Main analyse function ─────────────────────────────────────────────────────
async function analyseNews() {
  const text = textarea?.value?.trim();
  if (!text) {
    shakeTextarea();
    return;
  }
  if (text.length < 20) {
    shakeTextarea('Text is too short — please add more context.');
    return;
  }

  // Button loading state
  const btn = document.getElementById('analyseBtn');
  btn.querySelector('.btn-text').style.display = 'none';
  btn.querySelector('.btn-loading').style.display = 'inline-flex';
  btn.disabled = true;

  // Show progress
  showProgress();
  hideResult();

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || 'Server error');
    }

    const data = await response.json();
    hideProgress();
    displayResult(data, text);
  } catch (err) {
    hideProgress();
    displayError(err.message || 'Analysis failed. Please try again.');
  } finally {
    btn.querySelector('.btn-text').style.display = 'inline-flex';
    btn.querySelector('.btn-loading').style.display = 'none';
    btn.disabled = false;
  }
}

// ── Display result ────────────────────────────────────────────────────────────
function displayResult(data, originalText) {
  const panel = document.getElementById('resultPanel');
  const card = document.getElementById('resultCard');
  const verdictSection = document.getElementById('verdictSection');
  const verdictIcon = document.getElementById('verdictIcon');
  const verdictLabel = document.getElementById('verdictLabel');
  const verdictSubtitle = document.getElementById('verdictSubtitle');
  const gaugeFill = document.getElementById('gaugeFill');
  const gaugeValue = document.getElementById('gaugeValue');
  const realProb = document.getElementById('realProb');
  const fakeProb = document.getElementById('fakeProb');
  const signalsList = document.getElementById('signalsList');
  const wordCount = document.getElementById('wordCount');
  const readTime = document.getElementById('readTime');

  const isFake = data.label === 'FAKE';

  // ── Verdict ─────────────────────────────────────────────────────────────────
  verdictSection.className = 'verdict-section ' + (isFake ? 'is-fake' : 'is-real');

  verdictIcon.innerHTML = isFake
    ? '<i class="fas fa-circle-xmark" style="color:var(--neon-red)"></i>'
    : '<i class="fas fa-circle-check" style="color:var(--neon-green)"></i>';

  verdictLabel.innerHTML = isFake
    ? '<span class="fake-label">FAKE NEWS</span>'
    : '<span class="real-label">REAL NEWS</span>';

  const confidence = data.confidence;
  verdictSubtitle.textContent = isFake
    ? `${confidence}% confidence this is misinformation`
    : `${confidence}% confidence this is authentic news`;

  // ── Gauge ───────────────────────────────────────────────────────────────────
  gaugeFill.className = 'gauge-fill ' + (isFake ? 'fake-fill' : 'real-fill');
  setTimeout(() => {
    gaugeFill.style.width = confidence + '%';
  }, 200);

  // Animate gauge value
  let current = 0;
  const target = confidence;
  const duration = 1000;
  const startTime = performance.now();

  function animateGauge(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = eased * target;
    gaugeValue.textContent = current.toFixed(1) + '%';
    gaugeValue.style.color = isFake ? 'var(--neon-red)' : 'var(--neon-green)';
    if (progress < 1) requestAnimationFrame(animateGauge);
  }
  requestAnimationFrame(animateGauge);

  // ── Probabilities ────────────────────────────────────────────────────────────
  realProb.textContent = (data.real_prob * 100).toFixed(1) + '%';
  fakeProb.textContent = (data.fake_prob * 100).toFixed(1) + '%';

  // ── Signals ─────────────────────────────────────────────────────────────────
  signalsList.innerHTML = '';

  if (data.signals && data.signals.length > 0) {
    data.signals.forEach((sig, i) => {
      const li = document.createElement('li');
      li.className = 'signal-item';
      li.style.animationDelay = i * 80 + 'ms';
      li.innerHTML = `<i class="fas fa-triangle-exclamation"></i> ${escapeHtml(sig)}`;
      signalsList.appendChild(li);
    });
  } else {
    signalsList.innerHTML = '<li class="no-signals"><i class="fas fa-circle-check"></i> No obvious manipulation signals detected</li>';
  }

  // ── Stats ────────────────────────────────────────────────────────────────────
  const wc = originalText.split(/\s+/).filter(Boolean).length;
  wordCount.textContent = wc.toLocaleString();
  readTime.textContent = Math.ceil(wc / 200) + ' min';

  // ── Show panel ───────────────────────────────────────────────────────────────
  panel.style.display = 'block';
  panel.style.opacity = '0';
  panel.style.transform = 'translateY(30px)';

  requestAnimationFrame(() => {
    panel.style.transition = 'opacity 0.6s ease, transform 0.6s cubic-bezier(0.22, 1, 0.36, 1)';
    panel.style.opacity = '1';
    panel.style.transform = 'translateY(0)';
  });

  // Scroll to result
  setTimeout(() => {
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);

  // Card border glow
  card.style.borderColor = isFake ? 'rgba(248,113,113,0.3)' : 'rgba(52,211,153,0.3)';
  card.style.boxShadow = isFake
    ? '0 0 30px rgba(248,113,113,0.1)'
    : '0 0 30px rgba(52,211,153,0.1)';
}

// ── Copy result to clipboard ──────────────────────────────────────────────────
function copyResult() {
  const label = document.getElementById('verdictLabel')?.textContent?.trim();
  const confidence = document.getElementById('gaugeValue')?.textContent?.trim();
  const text = `SatyaCheck Analysis:\nVerdict: ${label}\nConfidence: ${confidence}\nPowered by SatyaCheck AI — satyacheck.ai`;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target.closest('button');
    const orig = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    btn.style.color = 'var(--neon-green)';
    setTimeout(() => {
      btn.innerHTML = orig;
      btn.style.color = '';
    }, 2000);
  });
}

// ── Display error ─────────────────────────────────────────────────────────────
function displayError(message) {
  const panel = document.getElementById('resultPanel');
  const card = document.getElementById('resultCard');

  card.innerHTML = `
    <div class="verdict-section" style="background:rgba(248,113,113,0.06);border:1px solid rgba(248,113,113,0.2);text-align:center;padding:2rem;border-radius:12px;">
      <div style="font-size:2.5rem;margin-bottom:1rem;"><i class="fas fa-triangle-exclamation" style="color:var(--neon-red)"></i></div>
      <div style="font-size:1.1rem;font-weight:600;margin-bottom:0.5rem;color:var(--neon-red)">Analysis Failed</div>
      <div style="color:var(--text-secondary);font-size:0.9rem;">${escapeHtml(message)}</div>
    </div>
    <div style="display:flex;justify-content:center;margin-top:1.5rem;">
      <button class="btn-ghost" onclick="clearInput()"><i class="fas fa-redo"></i> Try Again</button>
    </div>
  `;

  panel.style.display = 'block';
  panel.style.opacity = '1';
  panel.style.transform = 'none';
}

// ── Textarea shake animation ─────────────────────────────────────────────────
function shakeTextarea(msg) {
  if (!textarea) return;
  textarea.style.borderColor = 'var(--neon-red)';
  textarea.style.animation = 'shake 0.4s cubic-bezier(0.36,0.07,0.19,0.97) both';

  const style = document.createElement('style');
  style.textContent = '@keyframes shake{10%,90%{transform:translateX(-3px)}20%,80%{transform:translateX(4px)}30%,50%,70%{transform:translateX(-5px)}40%,60%{transform:translateX(5px)}}';
  document.head.appendChild(style);

  if (msg) {
    textarea.placeholder = msg;
    setTimeout(() => {
      textarea.placeholder = 'Paste news article, WhatsApp forward, tweet, or headline here...';
      textarea.style.borderColor = '';
    }, 2500);
  }

  textarea.addEventListener('animationend', () => {
    textarea.style.animation = '';
    textarea.style.borderColor = '';
  }, { once: true });
}

// ── Helper: escape HTML ───────────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Enter key shortcut ────────────────────────────────────────────────────────
textarea?.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    analyseNews();
  }
});
