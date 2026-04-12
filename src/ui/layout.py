"""
Scroll navigation and parallax helpers for the Streamlit app.

Uses window.parent.document body-append pattern so fixed elements are anchored
to the real viewport while the main Streamlit content keeps its normal layout.
"""

from __future__ import annotations

import streamlit.components.v1 as components

_SECTIONS = [
    ("sec-market", "Market Overview", 0.78, 1.00, "110,244,255"),
    ("sec-allocation", "Capacity Allocation", 0.92, 1.08, "0,184,200"),
    ("sec-demand", "Demand Map", 1.08, 1.14, "255,170,120"),
    ("sec-summary", "Summary", 0.88, 0.96, "160,155,255"),
    ("sec-sens", "Sensitivity", 1.18, 1.10, "255,230,109"),
]

_SECTIONS_JS = ",\n      ".join(
    (
        "{ "
        f"id: '{sec_id}', label: '{label}', speed: {speed}, depth: {depth}, accent: '{accent}'"
        " }"
    )
    for sec_id, label, speed, depth, accent in _SECTIONS
)

_INJECT_JS_TEMPLATE = """
<script>
(function() {
  try {
    var win = window.parent;
    var doc = win.document;
    var RETRY_LIMIT = 120;

    if (win.__onceScrollNav && typeof win.__onceScrollNav.destroy === 'function') {
      win.__onceScrollNav.destroy();
    }

    ['once-stars', 'once-scrollnav', 'once-overlaydeck'].forEach(function(id) {
      var el = doc.getElementById(id);
      if (el) el.remove();
    });

    var starWrap = doc.createElement('div');
    starWrap.id = 'once-stars';

    [
      [180, 1.0, [0.25, 0.65]],
      [80, 1.5, [0.35, 0.75]],
      [30, 2.0, [0.45, 0.90]]
    ].forEach(function(layer) {
      var count = layer[0];
      var size = layer[1];
      var range = layer[2];
      for (var i = 0; i < count; i += 1) {
        var star = doc.createElement('div');
        var opacity = range[0] + Math.random() * (range[1] - range[0]);
        var duration = 3.5 + Math.random() * 6.5;
        var delay = -(Math.random() * 10);
        var grey = Math.floor(200 + Math.random() * 55);
        star.style.cssText = [
          'position:absolute',
          'top:' + (Math.random() * 100).toFixed(3) + '%',
          'left:' + (Math.random() * 100).toFixed(3) + '%',
          'width:' + size + 'px',
          'height:' + size + 'px',
          'border-radius:50%',
          'background:rgb(' + grey + ',' + grey + ',' + grey + ')',
          '--star-op:' + opacity.toFixed(3),
          'opacity:' + opacity.toFixed(3),
          'animation:starTwinkle ' + duration.toFixed(2) + 's ' + delay.toFixed(2) + 's infinite ease-in-out',
          'pointer-events:none'
        ].join(';');
        starWrap.appendChild(star);
      }
    });
    doc.body.appendChild(starWrap);

    var sectionsCfg = [
      __SECTIONS__
    ];

    doc.querySelectorAll('.once-section-group, .once-section-source').forEach(function(group) {
      group.classList.remove('once-section-group', 'once-section-active', 'once-section-source');
      if (group.id && /-group$/.test(group.id)) {
        group.removeAttribute('id');
      }
      group.removeAttribute('data-sec');
      group.removeAttribute('data-label');
      group.removeAttribute('tabindex');
      group.style.removeProperty('--once-parallax-shift');
      group.style.removeProperty('--once-parallax-scale');
      group.style.removeProperty('--once-parallax-tilt');
      group.style.removeProperty('--once-parallax-opacity');
      group.style.removeProperty('--once-parallax-blur');
      group.style.removeProperty('--once-accent-rgb');
      group.style.removeProperty('--once-panel-lift');
      group.style.removeProperty('--once-source-opacity');
    });

    function findSectionRoot(anchor) {
      var host = anchor.closest('[data-testid="element-container"]');
      if (!host) return null;

      var block = host.parentElement;
      while (block && block !== doc.body && block.getAttribute('data-testid') !== 'stVerticalBlock') {
        block = block.parentElement;
      }
      if (!block) return null;

      return block.closest('[data-testid="stVerticalBlockBorderWrapper"]') || block;
    }

    function resolveSections() {
      var totalSections = sectionsCfg.length;
      return sectionsCfg.map(function(cfg, index) {
        var anchor = doc.getElementById(cfg.id);
        if (!anchor) return null;
        var root = findSectionRoot(anchor);
        if (!root) return null;

        root.classList.add('once-section-source');
        root.id = cfg.id + '-group';
        root.setAttribute('data-sec', cfg.id);
        root.setAttribute('data-label', cfg.label);
        root.setAttribute('tabindex', '-1');
        root.setAttribute('data-section-index', String(index));
        root.style.setProperty('--once-accent-rgb', cfg.accent);
        root.style.setProperty('--once-panel-lift', String(cfg.depth));
        root.style.setProperty('--once-panel-index', String(index));
        root.style.setProperty('--once-panel-order', String(totalSections - index));

        return {
          id: cfg.id,
          label: cfg.label,
          el: root,
          anchor: anchor,
          speed: cfg.speed || 1,
          depth: cfg.depth || 1,
          targetShift: 0,
          targetScale: 1,
          targetTilt: 0,
          targetOpacity: 1,
          targetBlur: 0,
          currentShift: 0,
          currentScale: 1,
          currentTilt: 0,
          currentOpacity: 1,
          currentBlur: 0
        };
      }).filter(Boolean);
    }

    function initScrollNav(sections) {
      var overlayDeck = doc.createElement('div');
      overlayDeck.id = 'once-overlaydeck';
      doc.body.appendChild(overlayDeck);

      var nav = doc.createElement('nav');
      nav.id = 'once-scrollnav';
      nav.setAttribute('aria-label', 'Section navigation');
      doc.body.appendChild(nav);

      var state = {
        activeIndex: 0,
        reducedMotion: win.matchMedia('(prefers-reduced-motion: reduce)').matches,
        ticking: false,
        springFrame: 0,
        ignoreHashChange: false,
        detachFns: []
      };

      sections.forEach(function(section, index) {
        var panel = doc.createElement('section');
        panel.className = 'once-section-group';
        panel.setAttribute('data-sec', section.id);
        panel.setAttribute('data-label', section.label);
        panel.setAttribute('data-overlay-index', String(index));
        panel.style.setProperty('--once-accent-rgb', section.el.style.getPropertyValue('--once-accent-rgb'));
        panel.style.setProperty('--once-panel-lift', section.el.style.getPropertyValue('--once-panel-lift'));
        panel.style.setProperty('--once-panel-index', section.el.style.getPropertyValue('--once-panel-index'));
        panel.style.setProperty('--once-panel-order', section.el.style.getPropertyValue('--once-panel-order'));

        var clone = section.el.cloneNode(true);
        clone.classList.remove('once-section-source');
        clone.removeAttribute('id');
        clone.removeAttribute('data-sec');
        clone.removeAttribute('data-label');
        clone.removeAttribute('data-section-index');
        clone.querySelectorAll('[id]').forEach(function(node) {
          node.removeAttribute('id');
        });
        clone.querySelectorAll('.once-section-anchor').forEach(function(anchor) {
          anchor.remove();
        });

        panel.appendChild(clone);
        overlayDeck.appendChild(panel);
        section.overlayEl = panel;
      });

      function addListener(target, eventName, handler, options) {
        target.addEventListener(eventName, handler, options);
        state.detachFns.push(function() {
          target.removeEventListener(eventName, handler, options);
        });
      }

      var pips = sections.map(function(section, index) {
        var pip = doc.createElement('button');
        pip.type = 'button';
        pip.className = 'snav-dot';
        pip.setAttribute('data-sec', section.id);
        pip.setAttribute('data-label', section.label);
        pip.setAttribute('title', section.label);
        pip.setAttribute('aria-label', 'Go to ' + section.label);
        pip.addEventListener('click', function() {
          goTo(index, true);
        });
        nav.appendChild(pip);
        return pip;
      });

      function updateHash(index, pushHistory) {
        var hash = '#' + sections[index].id;
        var url = new URL(win.location.href);
        url.hash = hash;
        state.ignoreHashChange = true;
        if (pushHistory) {
          win.history.pushState({ onceSectionIndex: index }, '', url);
        } else {
          win.history.replaceState({ onceSectionIndex: index }, '', url);
        }
        win.setTimeout(function() {
          state.ignoreHashChange = false;
        }, 0);
      }

      function setActive(index, pushHistory) {
        state.activeIndex = index;
        sections.forEach(function(section, i) {
          section.overlayEl.classList.toggle('once-section-active', i === index);
          section.el.style.setProperty('--once-source-opacity', i === index ? '0.08' : '0.02');
        });
        pips.forEach(function(pip, i) {
          var active = i === index;
          pip.classList.toggle('active', active);
          pip.setAttribute('aria-current', active ? 'true' : 'false');
        });
        updateHash(index, !!pushHistory);
      }

      function nearestIndex() {
        var viewportMid = win.innerHeight * 0.38;
        var bestIndex = 0;
        var bestDistance = Number.POSITIVE_INFINITY;
        sections.forEach(function(section, index) {
          var rect = section.el.getBoundingClientRect();
          var sectionMid = rect.top + Math.min(rect.height, win.innerHeight) * 0.28;
          var distance = Math.abs(sectionMid - viewportMid);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = index;
          }
        });
        return bestIndex;
      }

      function updateParallax() {
        state.ticking = false;
        var viewportHeight = Math.max(win.innerHeight, 1);

        sections.forEach(function(section) {
          var rect = section.el.getBoundingClientRect();
          var centerOffset = (rect.top + rect.height * 0.5 - viewportHeight * 0.5) / viewportHeight;
          var clamped = Math.max(-1.3, Math.min(1.3, centerOffset));
          var progress = 1 - Math.min(1, Math.abs(clamped));
          var direction = clamped >= 0 ? 1 : -1;
          section.targetShift = clamped * (-150 * section.speed);
          section.targetScale = 0.9 + progress * 0.12 - Math.abs(clamped) * 0.015 * section.depth;
          section.targetTilt = clamped * (10.5 * section.depth);
          section.targetOpacity = 0.34 + progress * 0.72;
          section.targetBlur = Math.max(0, (1 - progress) * 10.0);
          section.el.style.setProperty('--once-panel-progress', progress.toFixed(3));
          section.el.style.setProperty('--once-panel-direction', String(direction));
          section.overlayEl.style.setProperty('--once-panel-progress', progress.toFixed(3));
          section.overlayEl.style.setProperty('--once-panel-direction', String(direction));
        });

        var nextIndex = nearestIndex();
        if (nextIndex !== state.activeIndex) {
          setActive(nextIndex, false);
        }
        ensureSpring();
      }

      function animateSpring() {
        state.springFrame = 0;
        var stillMoving = false;
        var tension = state.reducedMotion ? 1.0 : 0.16;

        sections.forEach(function(section) {
          section.currentShift += (section.targetShift - section.currentShift) * tension;
          section.currentScale += (section.targetScale - section.currentScale) * tension;
          section.currentTilt += (section.targetTilt - section.currentTilt) * tension;
          section.currentOpacity += (section.targetOpacity - section.currentOpacity) * tension;
          section.currentBlur += (section.targetBlur - section.currentBlur) * tension;

          section.overlayEl.style.setProperty('--once-parallax-shift', section.currentShift.toFixed(2) + 'px');
          section.overlayEl.style.setProperty('--once-parallax-scale', section.currentScale.toFixed(3));
          section.overlayEl.style.setProperty('--once-parallax-tilt', section.currentTilt.toFixed(2) + 'deg');
          section.overlayEl.style.setProperty('--once-parallax-opacity', section.currentOpacity.toFixed(3));
          section.overlayEl.style.setProperty('--once-parallax-blur', section.currentBlur.toFixed(2) + 'px');

          if (
            Math.abs(section.targetShift - section.currentShift) > 0.18 ||
            Math.abs(section.targetScale - section.currentScale) > 0.001 ||
            Math.abs(section.targetTilt - section.currentTilt) > 0.03 ||
            Math.abs(section.targetOpacity - section.currentOpacity) > 0.004 ||
            Math.abs(section.targetBlur - section.currentBlur) > 0.04
          ) {
            stillMoving = true;
          }
        });

        if (stillMoving) {
          state.springFrame = win.requestAnimationFrame(animateSpring);
        }
      }

      function ensureSpring() {
        if (!state.springFrame) {
          state.springFrame = win.requestAnimationFrame(animateSpring);
        }
      }

      function requestParallaxUpdate() {
        if (state.ticking) return;
        state.ticking = true;
        win.requestAnimationFrame(updateParallax);
      }

      function goTo(index, pushHistory) {
        if (index < 0 || index >= sections.length) return;
        setActive(index, !!pushHistory);
        sections[index].el.scrollIntoView({
          behavior: state.reducedMotion ? 'auto' : 'smooth',
          block: 'start'
        });
        sections[index].el.focus({ preventScroll: true });
      }

      function handleHashChange() {
        if (state.ignoreHashChange) return;
        var hash = (win.location.hash || '').replace('#', '');
        var index = sections.findIndex(function(section) {
          return section.id === hash;
        });
        if (index >= 0) {
          goTo(index, false);
        }
      }

      function handleKeydown(event) {
        var target = event.target;
        if (target && target.closest('input, textarea, select, [contenteditable="true"]')) return;

        if (event.key === 'ArrowDown' || event.key === 'PageDown') {
          event.preventDefault();
          goTo(Math.min(sections.length - 1, state.activeIndex + 1), true);
        } else if (event.key === 'ArrowUp' || event.key === 'PageUp') {
          event.preventDefault();
          goTo(Math.max(0, state.activeIndex - 1), true);
        } else if (event.key === 'Home') {
          event.preventDefault();
          goTo(0, true);
        } else if (event.key === 'End') {
          event.preventDefault();
          goTo(sections.length - 1, true);
        }
      }

      addListener(win, 'scroll', requestParallaxUpdate, { passive: true });
      addListener(win, 'resize', requestParallaxUpdate, { passive: true });
      addListener(win, 'hashchange', handleHashChange, false);
      addListener(doc, 'keydown', handleKeydown, true);

      var initialIndex = sections.findIndex(function(section) {
        return section.id === (win.location.hash || '').replace('#', '');
      });
      if (initialIndex < 0) {
        initialIndex = nearestIndex();
      }
      setActive(initialIndex, false);
      requestParallaxUpdate();

      win.__onceScrollNav = {
        destroy: function() {
          state.detachFns.forEach(function(fn) { fn(); });
          state.detachFns = [];
          if (state.springFrame) {
            win.cancelAnimationFrame(state.springFrame);
            state.springFrame = 0;
          }
          if (overlayDeck && overlayDeck.parentNode) {
            overlayDeck.remove();
          }
        },
        goToSection: function(index) {
          goTo(index, true);
        }
      };
    }

    function bootstrap(attempt) {
      var sections = resolveSections();
      if (sections.length !== sectionsCfg.length) {
        if (attempt < RETRY_LIMIT) {
          win.setTimeout(function() {
            bootstrap(attempt + 1);
          }, 120);
        }
        return;
      }

      initScrollNav(sections);
    }

    bootstrap(0);
  } catch (error) {
    console.error('Once scroll nav init failed', error);
  }
})();
</script>
"""

_INJECT_JS = _INJECT_JS_TEMPLATE.replace("__SECTIONS__", _SECTIONS_JS)


def inject_scrollnav() -> None:
    """Inject the stars and fixed right-side navigation."""
    components.html(_INJECT_JS, height=0)


def section_start(sec_id: str, label: str) -> str:
    """Return a lightweight section anchor marker."""
    return (
        f'<div class="once-section-anchor" id="{sec_id}" data-sec="{sec_id}" tabindex="-1">'
        f'<div class="once-section-label">{label}</div>'
        f'<div class="once-section-sep"></div>'
        f'</div>'
    )
