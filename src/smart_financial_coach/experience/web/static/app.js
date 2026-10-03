// Theme: follows the system until the person picks one; the choice is kept in a cookie so the
// server renders it on the next page without a flash.
(function () {
  const root = document.documentElement;
  const systemDark = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.addEventListener("click", (event) => {
    if (!event.target.closest("[data-theme-toggle]")) return;
    const current = root.dataset.theme || (systemDark() ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    document.cookie = "theme=" + next + "; path=/; max-age=31536000; samesite=lax";
  });
  // Close the drill-down drawer with Escape or a click on the backdrop
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") document.getElementById("drawer")?.replaceChildren();
  });
  document.addEventListener("click", (event) => {
    if (event.target.classList?.contains("drawer") || event.target.closest("[data-close]")) {
      document.getElementById("drawer")?.replaceChildren();
    }
  });
  // Chat: clear the box as soon as a question is sent, and keep the newest answer in view
  document.addEventListener("htmx:beforeRequest", (event) => {
    const form = event.detail.elt;
    if (form.id === "ask-form") {
      const pending = document.getElementById("pending-q");
      if (pending) pending.textContent = form.question.value;
      form.querySelector("button[type=submit]").disabled = true;
    }
  });
  document.addEventListener("htmx:afterRequest", (event) => {
    const form = event.detail.elt;
    if (form.id === "ask-form") {
      form.reset();
      form.querySelector("button[type=submit]").disabled = false;
      form.question.focus();
    }
  });
  document.addEventListener("htmx:afterSwap", (event) => {
    if (event.detail.target.id === "turns") {
      event.detail.target.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  });
})();
