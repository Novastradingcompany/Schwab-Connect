document.addEventListener("DOMContentLoaded", () => {
  const parseNumber = (value) => {
    if (!value) {
      return Number.NEGATIVE_INFINITY;
    }
    const cleaned = value.replace(/[%$,]/g, "").trim();
    const parsed = parseFloat(cleaned);
    return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed;
  };

  document.querySelectorAll(".table-wrap").forEach((tableWrap) => {
    const refresh = parseInt(tableWrap.dataset.refresh || "0", 10);
    const columnToggleEnabled = tableWrap.dataset.columnToggle === "1";
    if (!Number.isNaN(refresh) && refresh > 0) {
      setInterval(() => window.location.reload(), refresh * 1000);
    }

    const table = tableWrap.querySelector("table");
    if (!table) {
      return;
    }

    const hideColumn = (index) => {
      const header = table.querySelectorAll("thead th")[index];
      if (!header) {
        return;
      }
      header.style.display = "none";
      table.querySelectorAll("tbody tr").forEach((row) => {
        const cell = row.children[index];
        if (cell) {
          cell.style.display = "none";
        }
      });
    };

    const showColumn = (index) => {
      const header = table.querySelectorAll("thead th")[index];
      if (!header) {
        return;
      }
      header.style.display = "";
      table.querySelectorAll("tbody tr").forEach((row) => {
        const cell = row.children[index];
        if (cell) {
          cell.style.display = "";
        }
      });
    };

    if (columnToggleEnabled) {
      const toggleBar = tableWrap.previousElementSibling?.classList.contains("column-toggle-bar")
        ? tableWrap.previousElementSibling
        : null;
      const hiddenColumns = new Set();

      const getVisibleCount = () =>
        Array.from(table.querySelectorAll("thead th")).filter((th) => th.style.display !== "none").length;

      const getHeaderLabel = (th, index) => {
        const text = (th.textContent || "").trim().replace(/\s+/g, " ");
        return text || `Column ${index + 1}`;
      };

      const renderToggleBar = () => {
        if (!toggleBar) {
          return;
        }
        toggleBar.innerHTML = "";
        if (hiddenColumns.size === 0) {
          const msg = document.createElement("span");
          msg.className = "column-toggle-empty";
          msg.textContent = "Click a column header to hide it.";
          toggleBar.appendChild(msg);
          return;
        }
        hiddenColumns.forEach((index) => {
          const header = table.querySelectorAll("thead th")[index];
          const button = document.createElement("button");
          button.type = "button";
          button.className = "column-toggle-chip";
          button.textContent = `Show ${getHeaderLabel(header, index)}`;
          button.addEventListener("click", () => {
            hiddenColumns.delete(index);
            showColumn(index);
            renderToggleBar();
          });
          toggleBar.appendChild(button);
        });
      };

      table.querySelectorAll("thead th").forEach((th, index) => {
        th.title = "Click to hide this column";
        th.addEventListener("click", () => {
          if (th.style.display === "none") {
            return;
          }
          if (getVisibleCount() <= 1) {
            return;
          }
          hiddenColumns.add(index);
          hideColumn(index);
          renderToggleBar();
        });
      });
      renderToggleBar();
    }

    if (!columnToggleEnabled) {
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
    }
  });
});
