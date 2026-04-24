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

  // ── Load index ──────────────────────────────────────────────────────────
  fetch(BASE + "index.json")
    .then(r => r.json())
    .then(data => {
      allHabs = data;

      const subjects = new Set();
      data.forEach(h => subjects.add(h.subject));
      Array.from(subjects).sort().forEach(s => {
        const opt = document.createElement("option");
        opt.value = s; opt.textContent = s;
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

    let results;
    if (!q) {
      results = allHabs
        .filter(h => (!stage || h.stage === stage) && (!subject || h.subject === subject))
        .slice(0, 100);
    } else {
      results = fuse.search(q)
        .map(r => r.item)
        .filter(h => (!stage || h.stage === stage) && (!subject || h.subject === subject))
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
        <span class="result-subject">${esc(h.subject)}</span>
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
