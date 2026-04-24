/* Detail page logic */
(function () {
  "use strict";

  const BASE = "data/habilidades/";

  const TIER_LABEL = {
    merge:          "Fusão",
    link_regional:  "Link regional",
    link:           "Link",
    none:           "Sem correspondência",
  };

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
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function habLink(code, descPt) {
    return `<span class="hab-link-code" onclick="navigate('${esc(code)}')">${esc(code)}</span>
            <span class="hab-link-desc">${esc(descPt)}</span>`;
  }

  function navigate(code) {
    window.location.href = `habilidade.html?code=${encodeURIComponent(code)}`;
  }
  window.navigate = navigate;

  function tierBadge(tier) {
    return `<span class="comp-tier tier-${esc(tier)}">${esc(TIER_LABEL[tier] || tier)}</span>`;
  }

  function scoreBar(score) {
    const pct = Math.round(score * 100);
    const fill = Math.min(100, Math.round(score * 333)); // scale: 0.3 = full bar
    return `
      <div class="cc-score-bar">
        <span class="cc-score-val">${pct}%</span>
        <div class="cc-score-track">
          <div class="cc-score-fill" style="width:${fill}%"></div>
        </div>
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
    html += `<div class="section">
      <div class="section-header">Componentes de aprendizagem</div>
      <div class="section-body">`;

    if (h.components.length === 0) {
      html += `<p class="prereq-obs">Nenhum componente gerado.</p>`;
    } else {
      html += `<ul class="comp-list">`;
      h.components.forEach(c => {
        html += `<li class="comp-item">
          ${tierBadge(c.match_tier)}
          <div class="comp-text">
            ${esc(c.description_pt)}
            ${c.cc_component_description
              ? `<div class="comp-cc">↔ <em>${esc(c.cc_component_description)}</em></div>`
              : ""}
          </div>
        </li>`;
      });
      html += `</ul>`;
    }
    html += `</div></div>`;

    // ── CC standard matches ────────────────────────────────────────────────
    html += `<div class="section">
      <div class="section-header">Correspondências com Common Core (Learning Commons)</div>
      <div class="section-body">`;

    if (h.cc_matches.length === 0) {
      html += `<p class="prereq-obs">Nenhuma correspondência com padrões Common Core encontrada.</p>`;
    } else {
      html += `<ul class="cc-list">`;
      h.cc_matches.forEach(m => {
        const tierParts = [];
        if (m.n_merge)         tierParts.push(`${m.n_merge} fusão`);
        if (m.n_link_regional) tierParts.push(`${m.n_link_regional} link regional`);
        if (m.n_link)          tierParts.push(`${m.n_link} link`);
        html += `<li class="cc-item">
          ${scoreBar(m.score)}
          <div class="cc-info">
            <span class="cc-code">${esc(m.standard_code)}</span>
            <div class="cc-desc">${esc(m.description)}</div>
            <div class="cc-tiers">${tierParts.join(" · ")}</div>
          </div>
        </li>`;
      });
      html += `</ul>`;
    }
    html += `</div></div>`;

    // ── Prerequisites ──────────────────────────────────────────────────────
    html += `<div class="section">
      <div class="section-header">Pré-requisitos</div>
      <div class="section-body">`;

    if (!isMath) {
      html += `<p class="prereq-obs">Obs: O Learning Commons ainda não mapeou pré-requisitos fora de Matemática.</p>`;
    } else if (h.prerequisites.length === 0) {
      html += `<p class="prereq-obs">Nenhum pré-requisito identificado para esta habilidade.</p>`;
    } else {
      html += `<ul class="hab-link-list">`;
      h.prerequisites.forEach(p => {
        html += `<li class="hab-link-item">${habLink(p.code, p.description_pt)}</li>`;
      });
      html += `</ul>`;
    }
    html += `</div></div>`;

    // ── Unlocks ────────────────────────────────────────────────────────────
    html += `<div class="section">
      <div class="section-header">Habilidades que esta desbloqueia</div>
      <div class="section-body">`;

    if (!isMath) {
      html += `<p class="prereq-obs">Obs: O Learning Commons ainda não mapeou pré-requisitos fora de Matemática.</p>`;
    } else if (h.unlocks.length === 0) {
      html += `<p class="prereq-obs">Nenhuma habilidade posterior identificada.</p>`;
    } else {
      html += `<ul class="hab-link-list">`;
      h.unlocks.forEach(u => {
        html += `<li class="hab-link-item">${habLink(u.code, u.description_pt)}</li>`;
      });
      html += `</ul>`;
    }
    html += `</div></div>`;

    // ── Render ─────────────────────────────────────────────────────────────
    contentEl.innerHTML = html;
    loadingEl.classList.add("hidden");
    contentEl.classList.remove("hidden");
  }
})();
