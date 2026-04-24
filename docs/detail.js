/* Detail page logic */
(function () {
  "use strict";

  const BASE = "data/habilidades/";

  const MATH_SUBJECTS = new Set([
    "Matemática",
    "Matemática e suas Tecnologias",
    "Espaços, Tempos, Quantidades, Relações e Transformações",
  ]);

  const loadingEl = document.getElementById("loading");
  const contentEl = document.getElementById("content");
  const errorEl   = document.getElementById("error");
  const titleEl   = document.getElementById("pageTitle");

  const params = new URLSearchParams(window.location.search);
  const code   = params.get("code");

  if (!code) {
    showError();
  } else {
    fetch(`${BASE}${encodeURIComponent(code)}.json`)
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(render)
      .catch(showError);
  }

  function showError() {
    loadingEl.classList.add("hidden");
    errorEl.classList.remove("hidden");
  }

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function habLink(code, descPt) {
    return `<span class="hab-link-code" onclick="navigate('${esc(code)}')">${esc(code)}</span>
            <span class="hab-link-desc">${esc(descPt)}</span>`;
  }

  function navigate(code) {
    window.location.href = `habilidade.html?code=${encodeURIComponent(code)}`;
  }
  window.navigate = navigate;

  function sourceBadge(tier) {
    if (tier === "merge") {
      return `<span class="comp-badge badge-lc">LC</span>`;
    }
    return `<span class="comp-badge badge-ai">AI</span>`;
  }

  function section(title, body) {
    return `<div class="section">
      <div class="section-header">${esc(title)}</div>
      <div class="section-body">${body}</div>
    </div>`;
  }

  function render(h) {
    titleEl.textContent = h.code;
    document.title = `${h.code} — BNCC × Learning Commons`;

    const isMath = MATH_SUBJECTS.has(h.subject);

    // ── Header card ────────────────────────────────────────────────────────
    let html = `
      <div class="hab-header">
        <div class="hab-code">${esc(h.code)}</div>
        <div class="hab-meta">${esc(h.stage)} · ${esc(h.grade)} · ${esc(h.subject)}</div>
        <div class="hab-description">${esc(h.description_pt)}</div>
      </div>
    `;

    // ── Components ─────────────────────────────────────────────────────────
    let compsBody;
    if (h.components.length === 0) {
      compsBody = `<p class="prereq-obs">Nenhum componente gerado.</p>`;
    } else {
      compsBody = `<ul class="comp-list">` +
        h.components.map(c => `
          <li class="comp-item">
            ${sourceBadge(c.match_tier)}
            <span class="comp-text">${esc(c.description_pt)}</span>
          </li>`).join("") +
        `</ul>`;
    }
    html += section("Componentes de aprendizagem", compsBody);

    // ── Prerequisites ──────────────────────────────────────────────────────
    let preBody;
    if (!isMath) {
      preBody = `<p class="prereq-obs">Obs: O Learning Commons ainda não mapeou pré-requisitos fora de Matemática.</p>`;
    } else if (h.prerequisites.length === 0) {
      preBody = `<p class="prereq-obs">Nenhum pré-requisito identificado para esta habilidade.</p>`;
    } else {
      preBody = `<ul class="hab-link-list">` +
        h.prerequisites.map(p => `<li class="hab-link-item">${habLink(p.code, p.description_pt)}</li>`).join("") +
        `</ul>`;
    }
    html += section("Pré-requisitos", preBody);

    // ── Unlocks ────────────────────────────────────────────────────────────
    let unlockBody;
    if (!isMath) {
      unlockBody = `<p class="prereq-obs">Obs: O Learning Commons ainda não mapeou pré-requisitos fora de Matemática.</p>`;
    } else if (h.unlocks.length === 0) {
      unlockBody = `<p class="prereq-obs">Nenhuma habilidade posterior identificada.</p>`;
    } else {
      unlockBody = `<ul class="hab-link-list">` +
        h.unlocks.map(u => `<li class="hab-link-item">${habLink(u.code, u.description_pt)}</li>`).join("") +
        `</ul>`;
    }
    html += section("Habilidades que esta desbloqueia", unlockBody);

    // ── Render ─────────────────────────────────────────────────────────────
    contentEl.innerHTML = html;
    loadingEl.classList.add("hidden");
    contentEl.classList.remove("hidden");
  }
})();
