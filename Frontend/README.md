Frontend landing page for hosting a LinkedIn-style "Zip" clone

This folder contains a minimal static landing page demonstrating how to host and share a "Zip" — a single ZIP file containing a user's resume, portfolio, and posts.

Files
- index.html — landing page
- styles.css — styles
- script.js — tiny interactivity
- logo.svg — simple logo

Quick start (serve locally)

1. From this folder run:

   python3 -m http.server 8000

2. Open http://localhost:8000 in your browser.

How to use
- Package a directory with your profile files into a ZIP named `yourname.zip` and place it in the static host or a public folder.
- Link to the ZIP file from your own page or use this site as a marketing/hosting landing page.

Notes
- This site is intentionally static and host-friendly (GitHub Pages, Netlify, S3, etc.).
- If you'd like, I can add example generator scripts to build a ZIP from a profile directory, or add CI config for automatic hosting (GitHub Pages).