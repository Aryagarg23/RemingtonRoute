Frontend landing page converted to a small Zip puzzle demo

This folder now contains a minimal static landing page demonstrating a small playable demo of "Zip" — the LinkedIn daily path-building puzzle. The demo is intentionally small and extendable.

Files
- index.html — landing page and demo UI
- styles.css — styles and game styles
- script.js — tiny interactivity (modal, year)
- zip-game.js — the playable Zip demo (one sample level)
- logo.svg — simple logo

Quick start (serve locally)

1. From this folder run:

   python3 -m http.server 8000

2. Open http://localhost:8000 in your browser and click "Play the demo" or the Play section.

How to play the demo
- Click or drag across the grid to visit adjacent cells.
- The path cannot revisit cells and must follow orthogonal adjacency.
- Numbered cells must be visited in ascending order when encountered.
- Click "Check solution" to validate the path; the demo checks full coverage and number order.

Notes & next steps
- The demo implements a tiny, client-side level (6x6 grid with a few numbered cells). It's a simple, educational implementation — not a complete faithful clone of LinkedIn's internal game logic.
- Next enhancements you might want: multiple levels, a solver, daily puzzles, better visuals/animations, mobile-friendly controls, and persistent progress.

Sources & references
- LinkedIn Games (Zip) and community writeups (search terms: "LinkedIn Zip game", "Zip puzzle")

If you'd like, I can now:
- Add more levels and a level selector.
- Implement a solver/backtracking helper to generate solvable levels.
- Add touch-friendly improvements and animations.
Tell me which and I'll implement the next step.