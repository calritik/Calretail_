"""
CalRetail — color system.

Sage-green / warm-off-white palette. Two groups that never mix:

  BRAND / chrome   — decorative only (sidebar, gradients, borders). Never data.
  CATEGORICAL /
  STATUS / RAMPS   — data encoding only.

Every value here has a CSS counterpart in assets/style.css and a mirror in
assets/chart_theme.js (which re-themes already-rendered Plotly figures when the
light/dark toggle fires, since a figure's colors are fixed at render time).
When you change a token, change it in all three places.
"""

# ── Ink & surfaces ────────────────────────────────────────────────────────────
INK         = "#25352b"   # deep forest charcoal — body text
INK_SOFT    = "#3a5442"
NAVY        = "#3f6652"   # repurposed as "deep sage" for gradients/badges
NAVY_DEEP   = "#243d2f"
SLATE_BG    = "#f7f6ef"   # warm off-white page background
CARD_LINE   = "#e2e4d7"
SURFACE     = "#ffffff"

# ── Brand ramp (chrome + primary series) ─────────────────────────────────────
BRAND   = "#5c8f6e"   # medium sage — primary accent, active states, links
BRAND2  = "#82ae8f"   # lighter sage — secondary fills, hover
BRAND3  = "#b7d4be"   # pale sage — soft highlights, dark-mode bright accent

# ── Status (fixed meaning — never reused as "series N") ──────────────────────
ACCENT      = "#c0724a"   # high / danger — muted terracotta
ACCENT_INK  = "#9c5934"
ACCENT_SOFT = "#efd0bc"
WARN        = "#d2a44e"   # medium — soft amber
WARN_INK    = "#8c6a2c"
OK          = "#4c8b63"   # low / healthy — leaf green (distinct hue from BRAND)
OK_INK      = "#35603f"
OK_SOFT     = "#bfe0c9"

STATUS = {
    "good":     OK,
    "warning":  WARN,
    "serious":  ACCENT_SOFT,
    "critical": ACCENT,
}

# Backend status/risk label -> pill class used by components.cards.pill().
# Anything unmapped falls through to the neutral pill, so a label that means
# "something is wrong" must be listed here or it will render as if it were fine.
RISK_CLASS = {
    # healthy
    "healthy": "low",   "low": "low",       "optimal": "low",     "good": "low",
    "in stock": "low",  "on track": "low",  "active": "low",
    # needs attention
    "watch": "medium",  "medium": "medium", "moderate": "medium", "warning": "medium",
    "underpriced": "medium",   # margin left on the table — worth a look, not urgent
    "overstock": "medium", "overstocked": "medium", "slow": "medium",
    # acting now
    "critical": "high", "high": "high",     "urgent": "high",     "at risk": "high",
    "overpriced": "high",      # priced above market: actively losing conversion
    "stockout": "high", "out of stock": "high", "breach": "high",
}

# ── Categorical (fixed order — never cycled) ─────────────────────────────────
# Sage-anchored so series 1 reads as "the brand series", then hue-separated for
# colorblind safety on both the light (#ffffff) and dark (#182620) card surface.
CATEGORICAL = [
    "#5c8f6e",  # 1 sage (brand)
    "#d2a44e",  # 2 amber
    "#c0724a",  # 3 terracotta
    "#8b7ba0",  # 4 dusty plum
    "#4e93a0",  # 5 teal
    "#a8a24e",  # 6 olive gold
    "#b0708a",  # 7 mauve
    "#b7d4be",  # 8 pale sage
]

# ── Sequential — single hue, light->dark (for a LIGHT card surface) ──────────
SEQUENTIAL_BLUE = [
    "#eef4f0", "#dcebe1", "#c3ddcd", "#a9cfb8", "#8fc0a3",
    "#77b090", "#5c8f6e", "#4a7659", "#3a5f47", "#2b4836",
]

# ── Diverging — terracotta (warm/bad) <-> sage (cool/good) ───────────────────
DIVERGING = ["#9c5934", "#c0724a", "#efd0bc", "#f0efe6", "#bfe0c9", "#82ae8f", "#5c8f6e"]


# ── Per-theme chart chrome ───────────────────────────────────────────────────
LIGHT = {
    "paper": "rgba(0,0,0,0)",
    "plot":  "rgba(0,0,0,0)",
    "ink":       INK,
    "ink_muted": "rgba(37,53,43,.55)",
    "grid":      "#eef2ea",
    "axis":      "#dde2d3",
    "tooltip_bg":   "#26362b",
    "tooltip_ink":  "#eef4ef",
    "tooltip_line": "rgba(183,212,190,.28)",
}

DARK = {
    "paper": "rgba(0,0,0,0)",
    "plot":  "rgba(0,0,0,0)",
    "ink":       "#cfe0d3",
    "ink_muted": "rgba(207,224,211,.55)",
    "grid":      "rgba(150,180,155,.14)",
    "axis":      "#2e4438",
    "tooltip_bg":   "#0b120d",
    "tooltip_ink":  "#eaf3ec",
    "tooltip_line": "rgba(140,190,150,.25)",
}


def chrome(dark: bool = False) -> dict:
    """Chart chrome for the active theme."""
    return DARK if dark else LIGHT


def risk_class(label: str | None) -> str:
    """Maps a backend risk/status string onto a .pill modifier class."""
    return RISK_CLASS.get((label or "").strip().lower(), "neutral")
