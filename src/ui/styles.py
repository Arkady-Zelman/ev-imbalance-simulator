"""
Once UI design tokens and app-wide CSS.
Inject ONCE_UI_CSS once at the top of app.py.
"""

from __future__ import annotations

ONCE_UI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600&display=swap');

:root {
    --brand-100: #d2fcff;
    --brand-300: #6ef4ff;
    --brand-500: #00b8c8;
    --brand-700: #007a87;
    --brand-900: #08002f;
    --page-bg: #050510;
    --surface: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --text-primary: #e8eaf0;
    --text-muted: #7a8090;
    --radius-m: 10px;
    --trans: 0.25s ease;
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"] {
    transform: none !important;
    will-change: auto !important;
    filter: none !important;
}

*, *::before, *::after { box-sizing: border-box; }

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
#root {
    background-color: var(--page-bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Geist', 'Inter', sans-serif !important;
}

body::before {
    content: '';
    display: block;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00b8c8, #6ef4ff, #00b8c8, transparent);
    z-index: 99999;
    pointer-events: none;
}

@keyframes starTwinkle {
    0%, 100% { opacity: var(--star-op, 0.6); }
    50% { opacity: calc(var(--star-op, 0.6) * 0.15); }
}

h1, .stApp h1 { font-size: 22px !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2, .stApp h2 { font-size: 17px !important; font-weight: 500 !important; }
h3, .stApp h3 { font-size: 14px !important; font-weight: 500 !important; }
p, li, .stMarkdown p { font-size: 12px !important; line-height: 1.6; }

label,
.stSelectbox label,
.stSlider label,
.stNumberInput label {
    font-size: 11px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

button[data-baseweb="tab"] { font-size: 12px !important; font-weight: 500; }

[data-testid="stSidebar"] {
    background: rgba(6,6,13,0.90) !important;
    backdrop-filter: blur(20px) saturate(1.4) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] { width: 270px !important; }

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

.stPlotlyChart,
.js-plotly-plot {
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

[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-m) !important;
    background: var(--surface) !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-m) !important;
    font-size: 12px !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--brand-700); }

.once-section-anchor {
    position: relative;
    margin: 0;
    padding: 0;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    transform: none !important;
    opacity: 1 !important;
}

.once-section-source {
    position: relative;
    margin: 0 0 36vh;
    opacity: var(--once-source-opacity, 0.03);
    filter: blur(1.6px) saturate(0.8);
    transform: scale(0.985);
    transform-origin: 50% 12%;
    transition: opacity 0.28s ease, filter 0.28s ease, transform 0.28s ease;
}

#once-overlaydeck {
    position: fixed;
    top: clamp(78px, 11vh, 120px);
    left: clamp(20px, 3vw, 56px);
    right: clamp(36px, 5vw, 92px);
    bottom: clamp(20px, 4vh, 40px);
    z-index: 30;
    pointer-events: none;
    perspective: 1400px;
    overflow: visible;
}

.once-section-group {
    position: absolute;
    inset: 0;
    margin: 0;
    padding: 1rem 1.15rem 1.7rem;
    min-height: 100%;
    border-radius: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    background:
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025)),
        radial-gradient(circle at top left, rgba(var(--once-accent-rgb, 110,244,255), 0.14), transparent 42%),
        rgba(7,8,20,0.60);
    backdrop-filter: blur(16px) saturate(1.25);
    -webkit-backdrop-filter: blur(16px) saturate(1.25);
    z-index: calc(20 + var(--once-panel-order, 1));
    transform:
        translateY(var(--once-parallax-shift, 0px))
        scale(var(--once-parallax-scale, 1))
        rotateX(var(--once-parallax-tilt, 0deg));
    opacity: var(--once-parallax-opacity, 1);
    transform-origin: 50% 12%;
    overflow: clip;
    isolation: isolate;
    will-change: transform, opacity, filter;
    filter: blur(var(--once-parallax-blur, 0px));
    transition:
        transform 0.34s ease,
        opacity 0.34s ease,
        filter 0.34s ease,
        border-color 0.28s ease,
        box-shadow 0.28s ease,
        background 0.28s ease;
    box-shadow:
        0 18px 48px rgba(0,0,0,0.20),
        inset 0 1px 0 rgba(255,255,255,0.04);
}

.once-section-group::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    pointer-events: none;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.06), transparent 26%),
        radial-gradient(circle at 14% 0%, rgba(var(--once-accent-rgb, 110,244,255), 0.18), transparent 34%);
    opacity: calc(0.25 + (var(--once-panel-progress, 0.5) * 0.55));
    z-index: 0;
}

.once-section-group::after {
    content: "";
    position: absolute;
    left: 6%;
    right: 6%;
    bottom: -22px;
    height: 42px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(0,0,0,0.26), rgba(0,0,0,0));
    transform: scaleX(calc(0.84 + (var(--once-panel-lift, 1) * 0.03)));
    opacity: calc(0.28 + (var(--once-panel-progress, 0.5) * 0.4));
    pointer-events: none;
    z-index: -1;
}

.once-section-group > div {
    position: relative;
    z-index: 1;
    height: 100%;
    overflow: hidden;
}

.once-section-group.once-section-active {
    border-color: rgba(var(--once-accent-rgb, 110,244,255), 0.28);
    box-shadow:
        0 28px 74px rgba(0,0,0,0.34),
        0 0 0 1px rgba(var(--once-accent-rgb, 110,244,255), 0.14),
        inset 0 1px 0 rgba(255,255,255,0.06);
}

.once-section-group [data-testid="stVerticalBlock"] {
    gap: 0.9rem;
}

.once-section-sep {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.16), transparent);
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

#once-scrollnav {
    position: fixed;
    right: clamp(10px, 1.4vw, 18px);
    top: 50%;
    transform: translateY(-50%);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 10px 8px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 999px;
    background: rgba(6,6,13,0.55);
    backdrop-filter: blur(18px) saturate(1.35);
    box-shadow: 0 10px 30px rgba(0,0,0,0.24);
}

.snav-dot {
    appearance: none;
    border: 0;
    display: block;
    width: 11px;
    height: 11px;
    padding: 0;
    border-radius: 999px;
    background: rgba(255,255,255,0.12);
    cursor: pointer;
    transition:
        background 0.28s ease,
        box-shadow 0.28s ease,
        transform 0.28s ease;
    position: relative;
}

.snav-dot.active,
.snav-dot:hover,
.snav-dot:focus-visible {
    background: var(--brand-300) !important;
    box-shadow:
        0 0 0 4px rgba(0,184,200,0.16),
        0 0 14px rgba(110,244,255,0.55) !important;
    transform: scale(1.18);
    outline: none;
}

.snav-dot::after {
    content: attr(data-label);
    position: absolute;
    right: 18px;
    top: 50%;
    transform: translateY(-50%);
    white-space: nowrap;
    font-size: 10px;
    font-family: 'Geist', sans-serif;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #e8eaf0;
    background: rgba(6,6,13,0.95);
    padding: 5px 9px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.22s ease, transform 0.22s ease;
}

.snav-dot:hover::after,
.snav-dot:focus-visible::after {
    opacity: 1;
    transform: translateY(-50%) translateX(-2px);
}

#once-stars {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

@media (prefers-reduced-motion: reduce) {
    .once-section-group,
    .snav-dot {
        transition: none !important;
        transform: none !important;
        opacity: 1 !important;
    }
}

@media (max-width: 900px) {
    #once-overlaydeck {
        top: 64px;
        left: 10px;
        right: 18px;
        bottom: 14px;
    }

    .once-section-source {
        margin-bottom: 26vh;
    }

    .once-section-group {
        min-height: auto;
        padding: 0.75rem 0.8rem 1.1rem;
        border-radius: 18px;
    }

    #once-scrollnav {
        right: 8px;
        gap: 10px;
        padding: 8px 6px;
    }
}
</style>
"""


PLOTLY_LAYOUT_BASE: dict = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist, Inter, sans-serif", size=11, color="#e8eaf0"),
    margin=dict(l=40, r=20, t=44, b=40),
)
