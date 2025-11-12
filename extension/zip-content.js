// Content script for Zip game pages.
// Auto-extracts the grid data on load and listens for solveSequence messages to dispatch clicks.

(function() {
  // helper to find grid div (mirrors template behavior)
  function getGridDiv(extractFromDocument) {
    let gridDiv = extractFromDocument(document);
    if (!gridDiv) {
      const frame = document.querySelector('iframe');
      if (frame) {
        try {
          const frameDoc = frame.contentDocument || frame.contentWindow.document;
          gridDiv = extractFromDocument(frameDoc);
        } catch (e) {
          // cross-origin - can't access
        }
      }
    }
    return gridDiv;
  }

  function doOneMouseCycle(clickTarget) {
    const commonClickArgs = { bubbles: true, cancelable: true, view: window};
    clickTarget.dispatchEvent(new MouseEvent('mousedown', commonClickArgs));
    clickTarget.dispatchEvent(new MouseEvent('mouseup', commonClickArgs));
  }

  function extractGameData() {
    const gridDiv = getGridDiv(d => d.querySelector('[data-testid="interactive-grid"]') || d.querySelector('.grid-game-board'));
    if (!gridDiv) return { error: 'grid-not-found' };

    // attempt to read --rows / --cols
    const rowsProp = gridDiv.style?.getPropertyValue('--rows')?.trim();
    const colsProp = gridDiv.style?.getPropertyValue('--cols')?.trim();
    let rows = null, cols = null;
    if (rowsProp && /^\d+$/.test(rowsProp)) rows = parseInt(rowsProp, 10);
    if (colsProp && /^\d+$/.test(colsProp)) cols = parseInt(colsProp, 10);
    if (!rows || !cols) {
      // try other style vars
      for (const p of Array.from(gridDiv.style || []).filter(p => p.startsWith('--'))) {
        const v = gridDiv.style.getPropertyValue(p).trim();
        if (/^\d+$/.test(v)) {
          rows = rows || parseInt(v, 10);
          cols = cols || parseInt(v, 10);
        }
      }
    }

    const children = Array.from(gridDiv.children || []);
    const numberedCells = [];
    const downWalls = [];
    const rightWalls = [];
    for (const child of children) {
      if (!child || !child.getAttribute) continue;
      const idxAttr = child.getAttribute('data-cell-idx');
      if (idxAttr == null) continue;
      const idx = parseInt(idxAttr, 10);
      // number content
      const contentEl = child.querySelector('[data-cell-content="true"]') || child.querySelector('.trail-cell-content');
      if (contentEl && contentEl.textContent && !isNaN(parseInt(contentEl.textContent, 10))) {
        const parsed = parseInt(contentEl.textContent, 10);
        numberedCells[parsed - 1] = idx;
      }
      if (child.querySelector('.trail-cell-wall--down')) downWalls.push(idx);
      if (child.querySelector('.trail-cell-wall--right')) rightWalls.push(idx);
    }

    return {
      rows: rows || null,
      cols: cols || null,
      numberedCells: numberedCells.filter(x => x !== undefined),
      downWalls,
      rightWalls
    };
  }

  // Immediately try to extract and notify the extension
  try {
    const data = extractGameData();
    chrome.runtime.sendMessage({ type: 'zipGameData', data });
  } catch (e) {
    console.error('zip-content extraction error', e);
    chrome.runtime.sendMessage({ type: 'zipGameData', data: { error: String(e) } });
  }

  // Listen for solve requests forwarded from the extension/service worker
  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (!msg || msg.type !== 'solveSequence') return;
    const sequence = msg.sequence || [];

    (async () => {
      const grid = getGridDiv(d => d.querySelector('[data-testid="interactive-grid"]') || d.querySelector('.grid-game-board'));
      if (!grid) {
        sendResponse({ ok: false, error: 'grid-not-found' });
        return;
      }

      function getCellElementByIdx(gridDiv, idx) {
        return Array.from(gridDiv.children || []).find(c => c && c.getAttribute && parseInt(c.getAttribute('data-cell-idx'), 10) === idx);
      }

      for (const idx of sequence) {
        const el = getCellElementByIdx(grid, idx);
        if (!el) {
          console.warn('No element for idx', idx);
          continue;
        }
        doOneMouseCycle(el);
        // small delay to allow UI to update
        await new Promise(r => setTimeout(r, 120));
      }
      sendResponse({ ok: true });
    })();

    // indicate async
    return true;
  });
})();
