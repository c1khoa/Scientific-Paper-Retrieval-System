const form = document.querySelector("#searchForm");
const queryInput = document.querySelector("#queryInput");
const methodSelect = document.querySelector("#methodSelect");
const topKSelect = document.querySelector("#topKSelect");
const statusText = document.querySelector("#statusText");
const resultsEl = document.querySelector("#results");

function truncate(text, maxLength = 360) {
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength).trim()}...` : text;
}

function formatScore(score) {
  if (typeof score !== "number") return "";
  return Number.isFinite(score) ? score.toFixed(4) : "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#039;");
}

function renderResults(results) {
  resultsEl.innerHTML = "";

  if (!results.length) {
    resultsEl.innerHTML = '<div class="empty-state">No papers found for this query.</div>';
    return;
  }

  const fragment = document.createDocumentFragment();

  results.forEach((paper) => {
    const item = document.createElement("article");
    item.className = "result-item";

    const authors = Array.isArray(paper.authors) ? paper.authors.slice(0, 4).join(", ") : "";
    const categories = Array.isArray(paper.categories) ? paper.categories.join(", ") : "";
    const score = formatScore(paper.score);
    const details = [
      authors && `<span>${escapeHtml(authors)}</span>`,
      categories && `<span class="pill">${escapeHtml(categories)}</span>`,
      score && `<span class="pill">score ${score}</span>`,
    ]
      .filter(Boolean)
      .join("");

    item.innerHTML = `
      <p class="result-url">${escapeHtml(paper.url)}</p>
      <a class="result-title" href="${escapeHtml(paper.url)}" target="_blank" rel="noopener noreferrer">
        ${escapeHtml(paper.title)}
      </a>
      <p class="result-summary">${escapeHtml(truncate(paper.abstract))}</p>
      <div class="result-details">${details}</div>
    `;

    fragment.appendChild(item);
  });

  resultsEl.appendChild(fragment);
}

function renderError(message) {
  resultsEl.innerHTML = `<div class="error-state">${escapeHtml(message)}</div>`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const query = queryInput.value.trim();
  if (!query) return;

  const submitButton = form.querySelector("button");
  submitButton.disabled = true;
  statusText.textContent = "Searching...";
  resultsEl.innerHTML = "";

  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        method: methodSelect.value,
        top_k: Number(topKSelect.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Search failed.");
    }

    statusText.textContent = `${data.count} results for "${data.query}" using ${data.method}.`;
    renderResults(data.results);
  } catch (error) {
    statusText.textContent = "Search could not complete.";
    renderError(error.message);
  } finally {
    submitButton.disabled = false;
  }
});
