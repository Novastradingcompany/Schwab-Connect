document.addEventListener("DOMContentLoaded", () => {
  const tableWrap = document.querySelector(".table-wrap");
  if (!tableWrap) {
    return;
  }
  const refresh = parseInt(tableWrap.dataset.refresh || "0", 10);
  if (!Number.isNaN(refresh) && refresh > 0) {
    setInterval(() => window.location.reload(), refresh * 1000);
  }
});
