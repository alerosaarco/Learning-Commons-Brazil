/* Search page logic */
(function () {
  "use strict";

  const BASE = "data/";
  let allHabs = [];
  let fuse = null;
  let subjects = new Set();

  const searchInput  = document.getElementById("searchInput");
  const searchMeta   = document.getElementById("searchMeta");
  const resultsList  = document.getElementById("results");
  const noResults    = document.getElementById("noResults");
  const filters      = document.getElementById("filters");
  const filterStage  = document.getElementById("filterStage");
  const filterSubject= document.getElementById("filterSubject");

  // ── Load index ──────────────────────────────────────────────────────────
  fetch(BASE + "index.json")
    .then(r => r.json())
    .then(data => {
      allHabs = data;

      data.forEach(h => subjects.add(h.subject));
      Array.from(subjects).sort().forEach(s => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
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
      searchMeta.textContent = `${data.length} habilidades carregadas`;
      render(allHabs.slice(0, 50));
    })
    .catch(() => {
      searchMeta.textContent = "Erro ao carregar dados.";
    });

  // ── Search ───────────────────────────────────────────────────────────────
  function currentFiltered() {
    const stage   = filterStage.value;
    const subject = filterSubject.value;
    let base = allHabs;
    if (stage)   base = base.filter(h => h.stage === stage);
    if (subject) base = base.filter(h => h.subject === subject);
    return base;
  }

  function doSearch() {
    const q      = searchInput.value.trim();
    const stage  = filterStage.value;
    const subject= filterSubject.value;

    let results;
    if (!q) {
      results = currentFiltered().slice(0, 80);
      searchMeta.textContent = results.length < (currentFiltered().length)
        ? `Mostrando 80 de ${currentFiltered().length} habilidades`
        : `${results.length} habilidades`;
    } else {
      // Fuse searches all, then filter
      const raw = fuse.search(q).map(r => r.item);
      results = raw.filter(h =>
        (!stage   || h.stage   === stage) &&
        (!subject || h.subject === subject)
      ).slice(0, 80);
      searchMeta.textContent = `${results.length} resultado(s) para "${q}"`;
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
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ── Events ───────────────────────────────────────────────────────────────
  let debounce;
  searchInput.addEventListener("input", () => {
    clearTimeout(debounce);
    debounce = setTimeout(doSearch, 200);
  });
  filterStage.addEventListener("change",   doSearch);
  filterSubject.addEventListener("change", doSearch);
})();
