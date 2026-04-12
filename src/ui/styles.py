"""
Once UI Magic Portfolio design tokens applied to Streamlit.
Inject ONCE_UI_CSS once at the top of app_v2.py.

Revert: switch back to `streamlit run app.py` — this file is only used by app_v2.py.
"""
from __future__ import annotations

ONCE_UI_CSS = """
<style>
/* ── Google Fonts: Geist ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&display=swap');

/* ── Design tokens ───────────────────────────────────────────────────── */
:root {
    --brand-100: #d2fcff;
    --brand-300: #6ef4ff;
    --brand-500: #00b8c8;
    --brand-700: #007a87;
    --brand-900: #08002f;
    --page-bg:   #050510;   /* deep space blue-black */
    --surface:   rgba(255,255,255,0.04);
    --border:    rgba(255,255,255,0.08);
    --text-primary: #e8eaf0;
    --text-muted:   #7a8090;
    --radius-m: 10px;
    --trans:    0.25s ease;
}

/* ── Reset Streamlit transform-based containing blocks ───────────────────
   Streamlit applies CSS transforms/will-change on its wrappers which creates
   a new CSS containing block, trapping position:fixed elements inside it.
   This override restores correct viewport-relative fixed positioning. */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"] {
    transform: none !important;
    will-change: auto !important;
    filter: none !important;
}

/* ── Base ─────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stApp"], #root {
    background-color: var(--page-bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Geist', 'Inter', sans-serif !important;
}
/* Visible accent line across the top — confirms CSS injection is working */
body::before {
    content: '';
    display: block;
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00b8c8, #6ef4ff, #00b8c8, transparent);
    z-index: 99999;
    pointer-events: none;
}

/* ── Starry sky keyframe (stars created by layout.py JS injection) ──────
   Stars use --star-op CSS custom property set inline per element.         */
@keyframes starTwinkle {
    0%,  100% { opacity: var(--star-op, 0.6); }
    50%       { opacity: calc(var(--star-op, 0.6) * 0.15); }
}

/* ── Typography ──────────────────────────────────────────────────────── */
h1, .stApp h1 { font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2, .stApp h2 { font-size: 17px !important; font-weight: 500 !important; }
h3, .stApp h3 { font-size: 14px !important; font-weight: 500 !important; }
p, li, .stMarkdown p { font-size: 12px !important; line-height: 1.6; }
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 11px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
button[data-baseweb="tab"] { font-size: 12px !important; font-weight: 500; }

/* ── Sidebar glass panel ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(6,6,13,0.90) !important;
    backdrop-filter: blur(20px) saturate(1.4) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] { width: 270px !important; }

/* ── Metric cards ────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-m);
    padding: 12px 16px;
    transition: background var(--trans), box-shadow var(--trans);
}
div[data-testid="metric-container"]:hover {
    background: rgba(0,184,200,0.06);
    box-shadow: 0 0 0 1px rgba(0,184,200,0.18);
}
[data-testid="stMetricValue"] { font-size: 1.5rem !important; }

/* ── Plotly chart — Windows Aero glass hover ─────────────────────────── */
.stPlotlyChart, .js-plotly-plot {
    border-radius: var(--radius-m) !important;
    transition: box-shadow var(--trans), background var(--trans) !important;
}
.stPlotlyChart:hover {
    backdrop-filter: blur(6px) saturate(1.6) !important;
    background: rgba(0,184,200,0.05) !important;
    box-shadow:
        0 8px 32px rgba(0,184,200,0.14),
        0 0 0 1px rgba(0,184,200,0.22) !important;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-m) !important;
    background: var(--surface) !important;
}

/* ── Alerts / info boxes ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-m) !important;
    font-size: 12px !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--brand-700); }

/* ── Section chrome ──────────────────────────────────────────────────── */
.once-section {
    padding-top: 8px;
    animation: onceFadeUp 0.45s ease both;
}
.once-section-sep {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 8px 0 20px;
}
.once-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
@keyframes onceFadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Right scroll-spy nav (elements created by JS, styled here) ──────── */
#once-scrollnav {
    position: fixed;
    right: 14px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 6px 2px;
}
.snav-dot {
    display: block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
    cursor: pointer;
    text-decoration: none;
    transition: background 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    position: relative;
}
.snav-dot.active {
    background: var(--brand-500) !important;
    box-shadow: 0 0 10px rgba(0,184,200,0.55) !important;
    transform: scale(1.35);
}
.snav-dot:hover {
    background: var(--brand-500) !important;
    box-shadow: 0 0 10px rgba(0,184,200,0.55) !important;
    transform: scale(1.35);
}
/* Label tooltip */
.snav-dot::after {
    content: attr(data-label);
    position: absolute;
    right: 14px;
    top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
    font-size: 10px;
    font-family: 'Geist', sans-serif;
    color: #e8eaf0;
    background: rgba(6,6,13,0.94);
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid rgba(255,255,255,0.08);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
}
.snav-dot:hover::after { opacity: 1; }

/* ── Star overlay container (created by JS) ───────────────────────────── */
#once-stars {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}
</style>
"""

# Shared base for all Plotly figures in app_v2.py — transparent backgrounds so
# the starfield background shows through and the aero glass CSS can work.
PLOTLY_LAYOUT_BASE: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, Inter, sans-serif", size=11, color="#e8eaf0"),
    margin=dict(l=40, r=20, t=44, b=40),
)
