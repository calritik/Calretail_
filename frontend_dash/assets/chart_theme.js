/*
 * Keeps Plotly figures in step with the light/dark toggle.
 *
 * Figures are built server-side, so their axis/legend/hover colours are fixed
 * at render time and cannot know which theme the browser is in. Left alone,
 * every chart keeps its light-mode near-black ink and disappears against a dark
 * card. Re-theming here — in the one place that actually knows the current
 * theme — avoids threading a `dark` flag through all 26 figure constructions
 * across seven page modules.
 *
 * Charts are also re-themed when new ones appear, because a Dash callback
 * replaces a dcc.Graph wholesale and the replacement arrives light-themed.
 */
(function () {
  "use strict";

  // Mirrors frontend_dash/theme/colors.py LIGHT/DARK — keep both in sync.
  var LIGHT = {
    ink: "#25352b",
    muted: "rgba(37,53,43,.55)",
    grid: "#eef2ea",
    axis: "#dde2d3",
    tip: "#26362b",
    tipLine: "rgba(183,212,190,.28)"
  };
  var DARK = {
    ink: "#cfe0d3",
    muted: "rgba(207,224,211,.58)",
    grid: "rgba(150,180,155,.14)",
    axis: "#2e4438",
    tip: "#0b120d",
    tipLine: "rgba(140,190,150,.25)"
  };

  function isDark() {
    return document.body.classList.contains("dark");
  }

  function themeOne(gd, c) {
    if (!window.Plotly || !gd || !gd.layout) return;

    var up = {
      "font.color": c.ink,
      "legend.font.color": c.muted,
      "hoverlabel.bgcolor": c.tip,
      "hoverlabel.bordercolor": c.tipLine,
      "xaxis.tickfont.color": c.muted,
      "yaxis.tickfont.color": c.muted,
      "xaxis.title.font.color": c.muted,
      "yaxis.title.font.color": c.muted,
      "xaxis.linecolor": c.axis,
      "yaxis.linecolor": c.axis
    };

    // Only repaint a gridline that the figure actually asked to show — writing
    // gridcolor onto a hidden grid would switch it on.
    if (gd.layout.xaxis && gd.layout.xaxis.showgrid) up["xaxis.gridcolor"] = c.grid;
    if (gd.layout.yaxis && gd.layout.yaxis.showgrid) up["yaxis.gridcolor"] = c.grid;

    // The route map is a MapLibre subplot — swap the tile style so the basemap
    // matches the surface instead of a light map glowing on a dark card.
    if (gd.layout.map) {
      up["map.style"] = (c === DARK) ? "carto-darkmatter" : "carto-positron";
    }

    // Radar (supplier scorecard) lives on a polar subplot, not x/y axes.
    if (gd.layout.polar) {
      up["polar.radialaxis.gridcolor"] = c.grid;
      up["polar.radialaxis.linecolor"] = c.grid;
      up["polar.radialaxis.tickfont.color"] = c.muted;
      up["polar.angularaxis.gridcolor"] = c.grid;
      up["polar.angularaxis.linecolor"] = c.grid;
      up["polar.angularaxis.tickfont.color"] = c.ink;
    }

    // Free-floating labels (route-map stop names, reference-line captions).
    if (gd.layout.annotations && gd.layout.annotations.length) {
      gd.layout.annotations.forEach(function (a, i) {
        // Coloured callouts are deliberate; only recolour the default ink ones.
        if (!a.font || !a.font.color || a.font.color === LIGHT.ink ||
            a.font.color === DARK.ink || a.font.color === LIGHT.muted ||
            a.font.color === DARK.muted) {
          up["annotations[" + i + "].font.color"] = c.muted;
        }
      });
    }

    try {
      window.Plotly.relayout(gd, up);
    } catch (e) {
      /* a figure mid-render can reject relayout; the next pass catches it */
    }
  }

  function applyAll() {
    var c = isDark() ? DARK : LIGHT;
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      themeOne(gd, c);
    });
  }

  var pending = null;
  function schedule() {
    if (pending) clearTimeout(pending);
    // Debounced: Dash mutates the tree many times per page swap, and Plotly
    // needs to have finished drawing before relayout will stick.
    pending = setTimeout(applyAll, 180);
  }

  function start() {
    schedule();
    // Theme flips are a class change on <body>.
    new MutationObserver(schedule).observe(document.body, {
      attributes: true,
      attributeFilter: ["class"]
    });
    // New/replaced graphs arrive as subtree changes.
    new MutationObserver(schedule).observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
