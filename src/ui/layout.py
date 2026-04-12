"""
Scroll-spy navigation, starry background, and section wrappers for app.py (Once UI).

Uses window.parent.document body-append pattern so that position:fixed elements
are anchored to the real viewport — bypassing Streamlit's container hierarchy.
Works on same-origin (localhost dev + Streamlit Cloud same-domain deploys).
Fails silently in cross-origin contexts (no visible errors).
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

_SECTIONS = [
    ("sec-market",  "Market & Allocation"),
    ("sec-demand",  "Demand Map"),
    ("sec-summary", "Summary"),
    ("sec-sens",    "Sensitivity"),
]

# ── JS injected via a zero-height iframe → appends to window.parent.document.body
# This bypasses Streamlit's container overflow/transform constraints.
_INJECT_JS = """
<script>
(function() {
  try {
    var doc = window.parent.document;

    /* ── Idempotent cleanup on Streamlit re-renders ─────────────────── */
    ['once-stars', 'once-scrollnav'].forEach(function(id) {
      var el = doc.getElementById(id);
      if (el) el.parentNode.removeChild(el);
    });

    /* ── Starry sky ─────────────────────────────────────────────────── */
    var starWrap = doc.createElement('div');
    starWrap.id = 'once-stars';
    /* Styles from ONCE_UI_CSS (#once-stars selector) apply automatically
       since the <style> is already in the parent document.              */

    // Three layers: [count, sizePx, [opMin, opMax]]
    var layers = [
      [180, 1.0,  [0.25, 0.65]],
      [80,  1.5,  [0.35, 0.75]],
      [30,  2.0,  [0.45, 0.90]],
    ];

    layers.forEach(function(layer) {
      var count = layer[0], size = layer[1], opRange = layer[2];
      for (var i = 0; i < count; i++) {
        var star  = doc.createElement('div');
        var top   = (Math.random() * 100).toFixed(3) + '%';
        var left  = (Math.random() * 100).toFixed(3) + '%';
        var op    = (opRange[0] + Math.random() * (opRange[1] - opRange[0])).toFixed(3);
        var dur   = (3.5 + Math.random() * 6.5).toFixed(2) + 's';
        var delay = -(Math.random() * 10).toFixed(2) + 's'; /* negative = already mid-cycle */
        var grey  = Math.floor(200 + Math.random() * 55);   /* 200–255 white/grey tones */
        star.style.cssText = [
          'position:absolute',
          'top:' + top,
          'left:' + left,
          'width:' + size + 'px',
          'height:' + size + 'px',
          'border-radius:50%',
          'background:rgb(' + grey + ',' + grey + ',' + grey + ')',
          '--star-op:' + op,
          'opacity:' + op,
          'animation:starTwinkle ' + dur + ' ' + delay + ' infinite ease-in-out',
          'pointer-events:none',
        ].join(';');
        starWrap.appendChild(star);
      }
    });

    doc.body.appendChild(starWrap);

    /* ── Scroll-spy nav ─────────────────────────────────────────────── */
    var sections_cfg = [
      ['sec-market',  'Market & Allocation'],
      ['sec-demand',  'Demand Map'],
      ['sec-summary', 'Summary'],
      ['sec-sens',    'Sensitivity'],
    ];

    var nav = doc.createElement('div');
    nav.id = 'once-scrollnav';
    /* #once-scrollnav styles come from ONCE_UI_CSS already in parent doc */

    sections_cfg.forEach(function(cfg) {
      var dot = doc.createElement('a');
      dot.href = '#' + cfg[0];
      dot.className = 'snav-dot';
      dot.setAttribute('data-sec', cfg[0]);
      dot.setAttribute('data-label', cfg[1]);
      dot.setAttribute('title', cfg[1]);
      nav.appendChild(dot);
    });

    doc.body.appendChild(nav);

    /* ── IntersectionObserver for active dot ────────────────────────── */
    var secEls = doc.querySelectorAll('.once-section[data-sec]');
    var dots   = nav.querySelectorAll('.snav-dot[data-sec]');

    if (secEls.length > 0 && dots.length > 0) {
      /* Activate first dot by default */
      dots[0].classList.add('active');

      var obs = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
          if (!entry.isIntersecting) return;
          var secId = entry.target.getAttribute('data-sec');
          dots.forEach(function(d) {
            if (d.getAttribute('data-sec') === secId) {
              d.classList.add('active');
            } else {
              d.classList.remove('active');
            }
          });
        });
      }, { threshold: 0.25, rootMargin: '-10% 0px -30% 0px' });

      secEls.forEach(function(s) { obs.observe(s); });
    }

  } catch(e) {
    /* Cross-origin / sandboxed env — graceful no-op */
  }
})();
</script>
"""


def inject_scrollnav() -> None:
    """
    Inject starry sky background + fixed scroll-spy nav into the parent document.
    Must be called after ONCE_UI_CSS is already injected (styles apply to injected elements).
    """
    components.html(_INJECT_JS, height=1)


def section_start(sec_id: str, label: str) -> str:
    """Return opening HTML for a scroll-spy section with fade-in animation."""
    return (
        f'<div class="once-section" id="{sec_id}" data-sec="{sec_id}">'
        f'<div class="once-section-label">{label}</div>'
        f'<div class="once-section-sep"></div>'
    )


def section_end() -> str:
    """Return closing HTML for a scroll-spy section."""
    return "</div>"
