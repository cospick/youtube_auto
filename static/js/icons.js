// Lucide 아이콘 부트스트랩 + 자동 리렌더
(function () {
  const script = document.createElement('script');
  script.src = 'https://unpkg.com/lucide@latest/dist/umd/lucide.min.js';
  script.onload = () => {
    if (!window.lucide) return;
    lucide.createIcons();

    let pending = false;
    const refresh = () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        lucide.createIcons();
      });
    };

    const observer = new MutationObserver((mutations) => {
      for (const m of mutations) {
        for (const node of m.addedNodes) {
          if (node.nodeType !== 1) continue;
          if (node.matches?.('[data-lucide]') || node.querySelector?.('[data-lucide]')) {
            refresh();
            return;
          }
        }
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });

    window.refreshIcons = refresh;
  };
  document.head.appendChild(script);
})();
