/* Search page logic */
(function () {
  "use strict";

  const BASE = "data/";
  let allHabs = [];
  let fuse = null;

  const searchInput   = document.getElementById("searchInput");
  const resultsList   = document.getElementById("results");
  const noResults     = document.getElementById("noResults");
  const filters       = document.getElementById("filters");
  const filterStage   = document.getElementById("filterStage");
  const filterSubject = document.getElementById("filterSubject");

  // Canonical discipline groups.
  // The dropdown shows only the label; filtering matches ALL subjects in the group.
  // This means "Matemática e suas Tecnologias" (EM) appears under "Matemática", etc.
  const SUBJECT_GROUPS = [
    { label: "Matemática", subjects: [
        "Matemática",
        "Matemática e suas Tecnologias",
        "Espaços, Tempos, Quantidades, Relações e Transformações",
    ]},
    { label: "Língua Portuguesa", subjects: [
        "Língua Portuguesa",
        "Linguagens e suas Tecnologias",
        "Escuta, Fala, Pensamento e Imaginação",
    ]},
    { label: "Língua Inglesa", subjects: [
        "Língua Inglesa",
    ]},
    { label: "Ciências", subjects: [
        "Ciências",
        "Ciências da Natureza e suas Tecnologias",
    ]},
    { label: "Ciências Humanas", subjects: [
        "História",
        "Geografia",
        "Ciências Humanas e Sociais Aplicadas",
        "O eu, o outro e o nós",
    ]},
    { label: "Arte", subjects: [
        "Arte",
        "Traços, Sons, Cores e Formas",
    ]},
    { label: "Educação Física", subjects: [
        "Educação Física",
        "Corpo, Gestos e Movimentos",
    ]},
    { label: "Ensino Religioso", subjects: ["Ensino Religioso"] },
    { label: "Computação",       subjects: ["Computação"] },
  ];

  // flat map: raw subject string → canonical group label
  const subjectToGroup = {};
  SUBJECT_GROUPS.forEach(g => g.subjects.forEach(s => { subjectToGroup[s] = g.label; }));

  // ── Load index ──────────────────────────────────────────────────────────
  fetch(BASE + "index.json")
    .then(r => r.json())
    .then(data => {
      allHabs = data;

      // Flat dropdown with canonical group labels, sorted alphabetically
      const present = new Set(data.map(h => h.subject));
      [...SUBJECT_GROUPS]
        .filter(g => g.subjects.some(s => present.has(s)))
        .sort((a, b) => a.label.localeCompare(b.label, "pt"))
        .forEach(group => {
          const opt = document.createElement("option");
          opt.value = group.label;
          opt.textContent = group.label;
          filterSubject.appendChild(opt);
        });

      fuse = new Fuse(data, {
        keys: [
          { name: "code",           weight: 2   },
          { name: "description_pt", weight: 1   },
          { name: "subject",        weight: 0.5 },
        ],
        threshold: 0.35,
        minMatchCharLength: 2,
        includeScore: true,
      });

      filters.classList.remove("hidden");
    })
    .catch(() => {
      resultsList.innerHTML = '<li style="color:var(--muted);padding:2rem;text-align:center">Erro ao carregar dados.</li>';
    });

  // ── Search ───────────────────────────────────────────────────────────────
  function hasActiveFilter() {
    return filterStage.value !== "" || filterSubject.value !== "";
  }

  function doSearch() {
    const q       = searchInput.value.trim();
    const stage   = filterStage.value;
    const subject = filterSubject.value;

    // Show nothing until the user types or picks a filter
    if (!q && !hasActiveFilter()) {
      resultsList.innerHTML = "";
      noResults.classList.add("hidden");
      return;
    }

    // subject filter value is a canonical group label — match all disciplines in that group
    const matchesSubject = h => !subject || subjectToGroup[h.subject] === subject;

    let results;
    if (!q) {
      results = allHabs
        .filter(h => (!stage || h.stage === stage) && matchesSubject(h))
        .slice(0, 100);
    } else {
      results = fuse.search(q)
        .map(r => r.item)
        .filter(h => (!stage || h.stage === stage) && matchesSubject(h))
        .slice(0, 100);
    }

    render(results);
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render(items) {
    resultsList.innerHTML = "";
    noResults.classList.toggle("hidden", items.length > 0);
    items.forEach(h => {
      const li = document.createElement("li");
      li.className = "result-item";
      li.innerHTML = `
        <span class="result-code">${esc(h.code)}</span>
        <span class="result-desc">${esc(h.description_pt)}</span>
        <span class="result-subject">${esc(subjectToGroup[h.subject] || h.subject)}</span>
      `;
      li.addEventListener("click", () => {
        window.location.href = `habilidade.html?code=${encodeURIComponent(h.code)}`;
      });
      resultsList.appendChild(li);
    });
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // ── Events ───────────────────────────────────────────────────────────────
  let debounce;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(doSearch, 180);
  });
  filterStage.addEventListener("change",   doSearch);
  filterSubject.addEventListener("change", doSearch);
})();
