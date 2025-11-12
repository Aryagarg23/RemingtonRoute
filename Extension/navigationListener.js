// navigationListener: injects scripts on SPA navigation and handles runtime messages.

const polyfilledBrowser = (typeof browser !== 'undefined') ? browser : chrome;
const gameScriptMap = {
  '/games/zip': 'zip-content.js',
};

polyfilledBrowser.webNavigation.onHistoryStateUpdated
    .addListener(({ tabId, url }) => {
      const path = new URL(url).pathname;
      for (const [prefix, script] of Object.entries(gameScriptMap)) {
        if (path.startsWith(prefix)) {
          polyfilledBrowser.scripting.executeScript({
            target: { tabId },
            files: [script],
          }).catch(err => console.error('Injection failed', err));
        }
      }
    });

// Store last-seen game data by tabId so other parts can query it.
const lastGameData = new Map();

polyfilledBrowser.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  try {
    if (!msg || !msg.type) return;
    if (msg.type === 'zipGameData') {
      // msg should include { tabId, data }
      const tid = sender.tab?.id ?? msg.tabId;
      if (tid != null) {
        lastGameData.set(tid, { data: msg.data, url: sender.tab?.url ?? msg.url, when: Date.now() });
      }
      // ack
      sendResponse({ ok: true });
      return true;
    }

    if (msg.type === 'getLatestZipData') {
      const tid = msg.tabId ?? (sender.tab && sender.tab.id) ?? null;
      if (tid != null && lastGameData.has(tid)) {
        sendResponse({ ok: true, data: lastGameData.get(tid) });
      } else {
        sendResponse({ ok: false });
      }
      return true;
    }

    if (msg.type === 'solveSequence') {
      // msg: { tabId?, sequence }
      const forwardTo = msg.tabId;
      const forward = (tabId) => {
        polyfilledBrowser.tabs.sendMessage(tabId, { type: 'solveSequence', sequence: msg.sequence })
          .then(() => sendResponse({ ok: true }))
          .catch(err => sendResponse({ ok: false, error: String(err) }));
      };
      if (forwardTo != null) {
        forward(forwardTo);
      } else {
        // forward to active tab
        polyfilledBrowser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
          if (tabs && tabs.length) forward(tabs[0].id);
          else sendResponse({ ok: false, error: 'no-active-tab' });
        }).catch(err => sendResponse({ ok: false, error: String(err) }));
      }
      return true; // indicate we'll call sendResponse asynchronously
    }
  } catch (e) {
    console.error('Error in navigationListener onMessage', e);
  }
});
