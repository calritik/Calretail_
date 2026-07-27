/*
 * Light/dark theme toggle.
 *
 * Owned entirely by the browser rather than a dcc.Store + callback pair: the
 * store's localStorage restore races Dash's initial callback, so on a fresh
 * page load the callback fired with the layout default and the persisted
 * choice was lost on every navigation. Reading localStorage directly has no
 * such ordering problem, and the class lands without a server round-trip.
 *
 * The initial class is applied by an inline <head> script (see index_string in
 * app.py) so the correct theme paints on the first frame; this file only keeps
 * the toggle working afterwards.
 */
(function () {
  "use strict";

  var KEY = "calretail-theme";

  function current() {
    try {
      return localStorage.getItem(KEY) === "dark" ? "dark" : "light";
    } catch (e) {
      return "light";
    }
  }

  function apply(theme) {
    document.body.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(KEY, theme);
    } catch (e) {
      /* private mode — the toggle still works for this session */
    }
  }

  apply(current());

  // Delegated: page_header rebuilds the button on every route change, so
  // binding straight to the node would go stale after the first navigation.
  document.addEventListener("click", function (ev) {
    var btn = ev.target && ev.target.closest && ev.target.closest("#theme-toggle");
    if (!btn) return;
    ev.preventDefault();
    apply(current() === "dark" ? "light" : "dark");
  });
})();
