document.addEventListener("DOMContentLoaded", () => {
  const tableWrap = document.querySelector(".table-wrap");
  if (!tableWrap) {
    return;
  }
  const refresh = parseInt(tableWrap.dataset.refresh || "0", 10);
  if (!Number.isNaN(refresh) && refresh > 0) {
    setInterval(() => window.location.reload(), refresh * 1000);
  }

  const table = tableWrap.querySelector("table");
  if (!table) {
    return;
  }

  const parseNumber = (value) => {
    if (!value) {
      return Number.NEGATIVE_INFINITY;
    }
    const cleaned = value.replace(/[%$,]/g, "").trim();
    const parsed = parseFloat(cleaned);
    return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed;
  };

  table.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const index = Array.from(th.parentElement.children).indexOf(th);
      const tbody = table.querySelector("tbody");
      if (!tbody) {
        return;
      }
      const rows = Array.from(tbody.querySelectorAll("tr"));
      const ascending = th.dataset.sortDir !== "asc";
      rows.sort((a, b) => {
        const aText = a.children[index]?.textContent || "";
        const bText = b.children[index]?.textContent || "";
        const aVal = parseNumber(aText);
        const bVal = parseNumber(bText);
        return ascending ? aVal - bVal : bVal - aVal;
      });
      rows.forEach((row) => tbody.appendChild(row));
      table.querySelectorAll("th.sortable").forEach((header) => {
        header.dataset.sortDir = "";
      });
      th.dataset.sortDir = ascending ? "asc" : "desc";
    });
  });
});
