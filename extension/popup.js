// Popup script: extract full page HTML from the active tab and let user copy/download/open it.
// Also handles game data extraction and sequence execution for Zip puzzles.

const extractBtn = document.getElementById('extractBtn');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');
const openBtn = document.getElementById('openBtn');
const rawHtml = document.getElementById('rawHtml');

function setControlsEnabled(enabled) {
	copyBtn.disabled = !enabled;
	downloadBtn.disabled = !enabled;
	openBtn.disabled = !enabled;
}

extractBtn.addEventListener('click', async () => {
	setControlsEnabled(false);
	rawHtml.value = 'Extracting...';
	try {
		const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
		if (!tab || !tab.id) {
			rawHtml.value = 'No active tab found.';
			return;
		}

		// Execute script in the page to get the full HTML
		const results = await chrome.scripting.executeScript({
			target: { tabId: tab.id },
			func: () => document.documentElement.outerHTML
		});

		const html = results?.[0]?.result ?? '';
		rawHtml.value = html;
		setControlsEnabled(!!html);
	} catch (err) {
		console.error(err);
		rawHtml.value = 'Error extracting HTML: ' + (err && err.message ? err.message : String(err));
	}
});

copyBtn.addEventListener('click', async () => {
	try {
		await navigator.clipboard.writeText(rawHtml.value);
		copyBtn.textContent = 'Copied';
		setTimeout(() => (copyBtn.textContent = 'Copy'), 1500);
	} catch (err) {
		alert('Copy failed: ' + err);
	}
});

downloadBtn.addEventListener('click', () => {
	const html = rawHtml.value || '';
	const blob = new Blob([html], { type: 'text/html' });
	const url = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = url;
	a.download = 'page.html';
	document.body.appendChild(a);
	a.click();
	a.remove();
	setTimeout(() => URL.revokeObjectURL(url), 10000);
});

openBtn.addEventListener('click', () => {
	const html = rawHtml.value || '';
	const blob = new Blob([html], { type: 'text/html' });
	const url = URL.createObjectURL(blob);
	// Open blob in a new tab where user can save via browser UI
	chrome.tabs.create({ url });
	setTimeout(() => URL.revokeObjectURL(url), 10000);
});

// initialize
setControlsEnabled(false);

// --- Game-data extraction / action execution UI bindings ---
const extractGameDataBtn = document.getElementById('extractGameDataBtn');
const runSequenceBtn = document.getElementById('runSequenceBtn');
const gameDataTa = document.getElementById('gameData');
const sequenceInput = document.getElementById('sequenceInput');

extractGameDataBtn.addEventListener('click', async () => {
	gameDataTa.value = 'Extracting game data...';
	try {
		const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
		if (!tab || !tab.id) {
			gameDataTa.value = 'No active tab found.';
			return;
		}

		const results = await chrome.scripting.executeScript({
			target: { tabId: tab.id },
			func: () => {
				// Try to locate the grid div in the page or in the first iframe.
				function findGridDocAndDiv() {
					const attempt = (d) => {
						return d.querySelector('[data-testid="interactive-grid"]')
								|| d.querySelector('.grid-game-board');
					};
					let grid = attempt(document);
					let doc = document;
					if (!grid) {
						const frame = document.querySelector('iframe');
						if (frame) {
							try {
								const fd = frame.contentDocument || frame.contentWindow.document;
								const g = attempt(fd);
								if (g) {
									return { grid: g, doc: fd };
								}
							} catch (e) {
								// cross-origin iframe -> can't access
							}
						}
					}
					if (grid) return { grid, doc };
					return { grid: null, doc: document };
				}

				function readCssNumericVar(gridDiv) {
					// Look for CSS custom properties that are numeric (e.g. --rows: 5)
					try {
						const props = Array.from(gridDiv.style).filter(p => p.startsWith('--'));
						for (const p of props) {
							const v = gridDiv.style.getPropertyValue(p).trim();
							if (/^\d+$/.test(v)) return parseInt(v, 10);
						}
					} catch (e) {}
					return null;
				}

				const { grid } = findGridDocAndDiv();
				if (!grid) {
					return { error: 'Grid element not found' };
				}

				// Determine rows/cols via style vars or fallbacks
				let rows = readCssNumericVar(grid) || null;
				let cols = null;
				// Try explicit --rows / --cols
				try {
					const r = grid.style?.getPropertyValue('--rows')?.trim();
					const c = grid.style?.getPropertyValue('--cols')?.trim();
					if (r && /^\d+$/.test(r)) rows = parseInt(r, 10);
					if (c && /^\d+$/.test(c)) cols = parseInt(c, 10);
				} catch (e) {}
				if (!cols) cols = rows;

				// Collect cellDivs (elements that represent cells)
				const children = Array.from(grid.children || []);
				const filtered = children.filter(c => c && (c.getAttribute && c.getAttribute('data-cell-idx') != null));
				const cellDivs = new Array(filtered.length);
				const numberedCells = [];
				const downWalls = [];
				const rightWalls = [];
				for (const cellDiv of filtered) {
					const idxAttr = cellDiv.getAttribute('data-cell-idx');
					const idx = parseInt(idxAttr, 10);
					cellDivs[idx] = true; // placeholder - we only need index list
					// number content
					const contentEl = cellDiv.querySelector('[data-cell-content="true"]') || cellDiv.querySelector('.trail-cell-content');
					if (contentEl && contentEl.textContent && !isNaN(parseInt(contentEl.textContent, 10))) {
						const parsed = parseInt(contentEl.textContent, 10);
						numberedCells[parsed - 1] = idx;
					}
					if (cellDiv.querySelector('.trail-cell-wall--down')) downWalls.push(idx);
					if (cellDiv.querySelector('.trail-cell-wall--right')) rightWalls.push(idx);
				}

				return {
					rows: rows || null,
					cols: cols || null,
					numberedCells: numberedCells.filter(x => x !== undefined),
					downWalls,
					rightWalls
				};
			}
		});

		const gameData = results?.[0]?.result;
		if (!gameData) {
			gameDataTa.value = 'No game data found: ' + JSON.stringify(results?.[0]?.result ?? null);
			runSequenceBtn.disabled = true;
			return;
		}
		gameDataTa.value = JSON.stringify(gameData, null, 2);
		runSequenceBtn.disabled = false;
	} catch (err) {
		console.error(err);
		gameDataTa.value = 'Error extracting game data: ' + (err && err.message ? err.message : String(err));
		runSequenceBtn.disabled = true;
	}
});

runSequenceBtn.addEventListener('click', async () => {
	let seq;
	try {
		seq = JSON.parse(sequenceInput.value || '[]');
		if (!Array.isArray(seq)) throw new Error('Sequence must be a JSON array');
	} catch (err) {
		alert('Invalid sequence JSON: ' + err.message);
		return;
	}
	try {
		const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
		if (!tab || !tab.id) {
			alert('No active tab');
			return;
		}
		// Execute sequence in page context. We dispatch mouse events to cell elements
		await chrome.scripting.executeScript({
			target: { tabId: tab.id },
			args: [seq],
			func: async (sequence) => {
				function findGrid() {
					const query = d => d.querySelector('[data-testid="interactive-grid"]') || d.querySelector('.grid-game-board');
					let g = query(document);
					if (!g) {
						const frame = document.querySelector('iframe');
						if (frame) {
							try {
								const fd = frame.contentDocument || frame.contentWindow.document;
								g = query(fd);
							} catch (e) { /* ignore cross-origin */ }
						}
					}
					return g;
				}

				function getCellElementByIdx(gridDiv, idx) {
					return Array.from(gridDiv.children || []).find(c => c && c.getAttribute && parseInt(c.getAttribute('data-cell-idx'), 10) === idx);
				}

				const grid = findGrid();
				if (!grid) throw new Error('Grid not found in page');

				// Helper to dispatch a click cycle
				function doOneMouseCycle(clickTarget) {
					const commonClickArgs = { bubbles: true, cancelable: true, view: window};
					clickTarget.dispatchEvent(new MouseEvent('mousedown', commonClickArgs));
					clickTarget.dispatchEvent(new MouseEvent('mouseup', commonClickArgs));
				}

				for (const idx of sequence) {
					const el = getCellElementByIdx(grid, idx);
					if (!el) {
						console.warn('No element found for idx', idx);
						continue;
					}
					doOneMouseCycle(el);
					// small delay so mutations and UI updates can happen
					await new Promise(r => setTimeout(r, 120));
				}
				return { success: true };
			}
		});
		alert('Sequence dispatched');
	} catch (err) {
		console.error(err);
		alert('Error dispatching sequence: ' + (err && err.message ? err.message : String(err)));
	}
});