// Apply saved theme immediately to prevent flashing
(function() {
  const savedTheme = localStorage.getItem('pygame-theme');
  if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.setAttribute('data-theme', 'light');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
  }
})();

document.addEventListener('DOMContentLoaded', () => {
  // Theme toggle
  const toggleBtn = document.getElementById('theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('pygame-theme', newTheme);
    });
  }

  // Header Nav Tabs (Option B: Learn & Examples vs API Reference)
  const tabBtns = document.querySelectorAll('.nav-tab');
  if (tabBtns.length) {
    function switchTab(tabId) {
      tabBtns.forEach(btn => {
        const isActive = btn.getAttribute('data-tab') === tabId;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });

      const panels = document.querySelectorAll('.header-tab-panel');
      panels.forEach(panel => {
        const isCurrent = panel.getAttribute('id') === `tab-panel-${tabId}`;
        panel.classList.toggle('active', isCurrent);
      });
    }

    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabId = btn.getAttribute('data-tab');
        switchTab(tabId);
      });
    });

    // Determine initial tab based on current page URL context
    const path = window.location.pathname;
    const isApiDoc = (path.includes('/ref/') && !path.includes('examples.html')) || path.includes('/referencias/');
    switchTab(isApiDoc ? 'reference' : 'learn');
  }

  // Dynamic active chip highlight on examples page
  if (window.location.pathname.includes('examples.html')) {
    const updateActiveExampleChip = () => {
      const hash = window.location.hash;
      const exampleLinks = document.querySelectorAll('#tab-panel-learn .module-group:nth-child(2) .module-chip');
      exampleLinks.forEach(link => link.classList.remove('active'));

      if (hash) {
        const targetChip = document.querySelector(`#tab-panel-learn a[href$="${hash}"]`);
        if (targetChip) {
          targetChip.classList.add('active');
          return;
        }
      }

      const overviewChip = document.getElementById('chip-overview-all');
      if (overviewChip) {
        overviewChip.classList.add('active');
      }
    };

    updateActiveExampleChip();
    window.addEventListener('hashchange', updateActiveExampleChip);

    // Also update on chip click directly
    document.querySelectorAll('#tab-panel-learn .module-group:nth-child(2) .module-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        setTimeout(updateActiveExampleChip, 50);
      });
    });
  }
});
