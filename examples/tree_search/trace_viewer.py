#!/usr/bin/env python
"""Serve an interactive viewer for tree-search trace directories.

Example:

```bash
uv run python examples/tree_search/trace_viewer.py \
    --trace_dir outputs/tree_search/mcts_task0_k15 \
    --host 0.0.0.0 \
    --port 7860
```
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tree Search Trace</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1f242b;
      --muted: #667085;
      --line: #d6dae1;
      --line-strong: #aeb6c2;
      --blue: #2563eb;
      --green: #138a5b;
      --amber: #b7791f;
      --red: #c24136;
      --purple: #7c3aed;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, input, select {
      font: inherit;
    }

    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
    }

    button:hover { border-color: var(--line-strong); }
    button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }

    input, select {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
      min-width: 0;
    }

    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 10px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
      z-index: 2;
    }

    .title-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      min-width: 0;
    }

    h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2f7;
      color: #344054;
      white-space: nowrap;
      font-size: 12px;
    }

    .pill.success { background: #def7ec; color: #096c47; }
    .pill.fail { background: #fde7e5; color: #9f2f26; }

    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      flex-wrap: wrap;
    }

    .toolbar input { width: 190px; }
    .toolbar select { width: 150px; }

    .workspace {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(460px, 1fr) 420px;
    }

    .tree-pane {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border-right: 1px solid var(--line);
      background: #fff;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(90px, 1fr));
      gap: 1px;
      background: var(--line);
      border-bottom: 1px solid var(--line);
    }

    .metric {
      min-width: 0;
      background: #fff;
      padding: 8px 10px;
    }

    .metric .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .metric .value {
      margin-top: 2px;
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .canvas-wrap {
      min-height: 0;
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(#eef1f5 1px, transparent 1px),
        linear-gradient(90deg, #eef1f5 1px, transparent 1px);
      background-size: 28px 28px;
    }

    svg {
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
      user-select: none;
    }

    svg.dragging { cursor: grabbing; }

    .edge {
      fill: none;
      stroke: #b7c0cc;
      stroke-width: 2;
    }

    .edge.selected { stroke: var(--blue); stroke-width: 3; }

    .node {
      cursor: pointer;
    }

    .node-hit {
      fill: transparent;
      cursor: pointer;
      pointer-events: all;
    }

    .node circle {
      stroke: #fff;
      stroke-width: 2.5;
      filter: drop-shadow(0 1px 1px rgba(16, 24, 40, 0.25));
    }

    .node.selected circle {
      stroke: var(--blue);
      stroke-width: 4;
    }

    .node text {
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 4px;
      stroke-linejoin: round;
      fill: #1f2937;
      font-size: 12px;
      cursor: pointer;
      pointer-events: auto;
    }

    .legend {
      position: absolute;
      left: 12px;
      bottom: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 8px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      color: var(--muted);
      font-size: 12px;
    }

    .legend span {
      display: inline-flex;
      gap: 6px;
      align-items: center;
    }

    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
    }

    .details {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      background: var(--bg);
    }

    .details-inner {
      padding: 12px;
      display: grid;
      gap: 12px;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
    }

    .panel-body {
      padding: 12px;
    }

    .image-box {
      background: #111827;
      display: grid;
      place-items: center;
      min-height: 220px;
    }

    .image-box img {
      max-width: 100%;
      width: 100%;
      display: block;
      object-fit: contain;
    }

    .vlm-gallery {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 8px;
    }

    .vlm-summary {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 10px;
    }

    .vlm-section + .vlm-section {
      margin-top: 12px;
    }

    .vlm-section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .vlm-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
      cursor: zoom-in;
    }

    .vlm-card.reference { border-color: #b7a4e8; }
    .vlm-card.state { border-color: #9cc7bc; }
    .vlm-card.other { border-color: var(--line); }

    .vlm-card img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      display: block;
      background: #111827;
    }

    .vlm-card div {
      padding: 6px 7px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .vlm-card strong {
      display: block;
      color: var(--ink);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .vlm-card span {
      display: block;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .lightbox {
      position: fixed;
      inset: 0;
      z-index: 10;
      display: none;
      grid-template-rows: auto 1fr;
      background: rgba(12, 18, 31, 0.86);
      color: #fff;
    }

    .lightbox.open { display: grid; }

    .lightbox-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.18);
    }

    .lightbox-title {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 700;
    }

    .lightbox-body {
      min-height: 0;
      display: grid;
      place-items: center;
      padding: 16px;
    }

    .lightbox img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      background: #111827;
    }

    .kv {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .kv div {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 8px;
      background: #fbfcfe;
    }

    .kv span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .kv strong {
      display: block;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }

    .text-block, pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 0;
      color: #2e3642;
    }

    pre {
      max-height: 300px;
      overflow: auto;
      background: #101828;
      color: #e5e7eb;
      padding: 10px;
      border-radius: 6px;
      font-size: 12px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
    }

    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
      font-size: 12px;
    }

    th {
      color: var(--muted);
      font-weight: 600;
      background: #fbfcfe;
    }

    tr.clickable {
      cursor: pointer;
    }

    tr.clickable:hover {
      background: #eef6ff;
    }

    .steps {
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 2px;
    }

    .step {
      width: 96px;
      flex: 0 0 96px;
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      background: #fff;
    }

    .step img {
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      display: block;
      background: #111827;
    }

    .step div {
      padding: 5px 6px;
      font-size: 11px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .empty {
      color: var(--muted);
      padding: 16px;
      text-align: center;
    }

    .error {
      color: #9f2f26;
      background: #fde7e5;
      border: 1px solid #f4b4ae;
      border-radius: 8px;
      padding: 10px;
    }

    @media (max-width: 980px) {
      .topbar { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .workspace { grid-template-columns: 1fr; grid-template-rows: minmax(430px, 58vh) 1fr; }
      .tree-pane { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="title-row">
        <h1>Tree Search Trace</h1>
        <span id="trace-status" class="pill">loading</span>
        <span id="task-pill" class="pill"></span>
      </div>
      <div class="toolbar">
        <input id="node-search" type="search" placeholder="Node id, source, reason">
        <select id="depth-filter" aria-label="Depth"></select>
        <button id="fit-btn">Fit</button>
        <button id="reload-btn" class="primary">Reload</button>
      </div>
    </header>

    <main class="workspace">
      <section class="tree-pane">
        <div id="metrics" class="metrics"></div>
        <div class="canvas-wrap">
          <svg id="tree-svg" role="img" aria-label="Tree trace"></svg>
          <div class="legend">
            <span><i class="dot" style="background:#c24136"></i>0.0</span>
            <span><i class="dot" style="background:#b7791f"></i>0.5</span>
            <span><i class="dot" style="background:#138a5b"></i>1.0</span>
            <span><i class="dot" style="background:#7c3aed"></i>success</span>
          </div>
        </div>
      </section>

      <aside class="details">
        <div id="details" class="details-inner"></div>
      </aside>
    </main>
    <div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="VLM image preview">
      <div class="lightbox-head">
        <div id="lightbox-title" class="lightbox-title"></div>
        <button id="lightbox-close" type="button">Close</button>
      </div>
      <div class="lightbox-body">
        <img id="lightbox-image" alt="">
      </div>
    </div>
  </div>

  <script>
    const svg = document.getElementById("tree-svg");
    const details = document.getElementById("details");
    const metrics = document.getElementById("metrics");
    const statusPill = document.getElementById("trace-status");
    const taskPill = document.getElementById("task-pill");
    const searchInput = document.getElementById("node-search");
    const depthFilter = document.getElementById("depth-filter");
    const reloadBtn = document.getElementById("reload-btn");
    const fitBtn = document.getElementById("fit-btn");
    const lightbox = document.getElementById("lightbox");
    const lightboxTitle = document.getElementById("lightbox-title");
    const lightboxImage = document.getElementById("lightbox-image");
    const lightboxClose = document.getElementById("lightbox-close");

    const state = {
      tree: null,
      nodes: [],
      edges: [],
      nodeById: new Map(),
      edgesByParent: new Map(),
      edgeByChild: new Map(),
      selectedId: null,
      depth: "all",
      query: "",
      zoom: 1,
      panX: 40,
      panY: 40,
      dragging: false,
      dragStart: null,
      layoutBounds: { width: 800, height: 600 },
    };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      })[ch]);
    }

    function short(value, max = 90) {
      const text = String(value ?? "");
      return text.length > max ? text.slice(0, max - 1) + "..." : text;
    }

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "-";
      const num = Number(value);
      if (Number.isFinite(num)) return num.toFixed(digits).replace(/\.?0+$/, "");
      return String(value);
    }

    function imageUrl(path) {
      return path ? "/trace/" + path.split("/").map(encodeURIComponent).join("/") : "";
    }

    function nodeScore(node) {
      return node && (node.vlm_score ?? node.score ?? 0);
    }

    function nodeReason(node) {
      return node && (node.vlm_reason ?? node.score_reason ?? "");
    }

    function nodeMetadata(node) {
      return node && (node.vlm_metadata ?? node.score_metadata ?? null);
    }

    function nodeImages(node) {
      return node && (node.vlm_images ?? node.score_images ?? []);
    }

    function adaptiveNoise(value) {
      if (!value) return null;
      if (value.adaptive_noise) return value.adaptive_noise;
      if (value.parent_adaptive_noise) return value.parent_adaptive_noise;
      return null;
    }

    function scoreColor(score, success) {
      if (success) return "#7c3aed";
      const value = Math.max(0, Math.min(1, Number(score) || 0));
      if (value < 0.34) return "#c24136";
      if (value < 0.67) return "#b7791f";
      return "#138a5b";
    }

    function statusClass(success) {
      if (success === true) return "pill success";
      if (success === false) return "pill fail";
      return "pill";
    }

    async function loadTree() {
      statusPill.className = "pill";
      statusPill.textContent = "loading";
      try {
        const response = await fetch("/api/tree?ts=" + Date.now());
        if (!response.ok) throw new Error(await response.text());
        const tree = await response.json();
        hydrate(tree);
        renderAll();
        statusPill.className = statusClass(tree.summary && tree.summary.success);
        statusPill.textContent = tree.summary && tree.summary.success ? "success" : "open";
      } catch (error) {
        statusPill.className = "pill fail";
        statusPill.textContent = "error";
        details.innerHTML = `<div class="error">${esc(error.message)}</div>`;
        svg.replaceChildren();
      }
    }

    function hydrate(tree) {
      state.tree = tree;
      state.nodes = Array.isArray(tree.nodes) ? tree.nodes : [];
      state.edges = Array.isArray(tree.edges) ? tree.edges : [];
      state.nodeById = new Map(state.nodes.map((node) => [node.id, node]));
      state.edgesByParent = new Map();
      state.edgeByChild = new Map();

      for (const node of state.nodes) {
        node._children = [];
        node._match = true;
      }

      for (const edge of state.edges) {
        if (!state.edgesByParent.has(edge.parent_id)) state.edgesByParent.set(edge.parent_id, []);
        state.edgesByParent.get(edge.parent_id).push(edge);
        state.edgeByChild.set(edge.child_id, edge);
        const parent = state.nodeById.get(edge.parent_id);
        const child = state.nodeById.get(edge.child_id);
        if (parent && child && !parent._children.includes(child)) parent._children.push(child);
      }

      for (const node of state.nodes) {
        node._children.sort((a, b) => {
          const edgeA = state.edgeByChild.get(a.id) || {};
          const edgeB = state.edgeByChild.get(b.id) || {};
          return (edgeA.candidate_index ?? 0) - (edgeB.candidate_index ?? 0)
            || Number(nodeScore(b) || 0) - Number(nodeScore(a) || 0)
            || String(a.id).localeCompare(String(b.id));
        });
      }

      if (!state.selectedId || !state.nodeById.has(state.selectedId)) {
        state.selectedId = state.nodes[0] ? state.nodes[0].id : null;
      }

      updateDepthFilter();
    }

    function updateDepthFilter() {
      const depths = [...new Set(state.nodes.map((node) => Number(node.depth) || 0))]
        .sort((a, b) => a - b);
      const current = depthFilter.value || state.depth;
      depthFilter.innerHTML = `<option value="all">All depths</option>` +
        depths.map((depth) => `<option value="${depth}">Depth ${depth}</option>`).join("");
      depthFilter.value = depths.some((depth) => String(depth) === current) ? current : "all";
      state.depth = depthFilter.value;
    }

    function computeMatches() {
      const query = state.query.trim().toLowerCase();
      for (const node of state.nodes) {
        const depthOk = state.depth === "all" || String(node.depth) === state.depth;
        const haystack = [
          node.id,
          node.parent_id,
          node.planner,
          node.source,
          nodeReason(node),
          node.prompt,
          JSON.stringify(adaptiveNoise(node) || {}),
          node.success ? "success" : "",
          node.terminal ? "terminal" : "",
        ].join(" ").toLowerCase();
        node._match = depthOk && (!query || haystack.includes(query));
      }
    }

    function layoutTree() {
      computeMatches();
      const roots = state.nodes.filter((node) => !node.parent_id || !state.nodeById.has(node.parent_id));
      if (!roots.length && state.nodes.length) roots.push(state.nodes[0]);

      const depthGap = 220;
      const rowGap = 86;
      let nextY = 0;
      const visited = new Set();

      function visit(node) {
        if (visited.has(node.id)) return node._y || 0;
        visited.add(node.id);
        const children = node._children || [];
        node._x = (Number(node.depth) || 0) * depthGap;
        if (!children.length) {
          node._y = nextY * rowGap;
          nextY += 1;
        } else {
          const ys = children.map(visit);
          node._y = ys.reduce((sum, value) => sum + value, 0) / ys.length;
        }
        return node._y;
      }

      for (const root of roots) {
        visit(root);
        nextY += 0.45;
      }

      for (const node of state.nodes) {
        if (!visited.has(node.id)) visit(node);
      }

      const maxDepth = Math.max(0, ...state.nodes.map((node) => Number(node.depth) || 0));
      state.layoutBounds = {
        width: Math.max(640, maxDepth * depthGap + 260),
        height: Math.max(360, nextY * rowGap + 120),
      };
    }

    function renderAll() {
      const summary = state.tree.summary || {};
      taskPill.textContent = summary.task || `${summary.suite || "trace"} ${summary.task_id ?? ""}`;
      renderMetrics();
      layoutTree();
      renderTree();
      renderDetails();
    }

    function renderMetrics() {
      const summary = state.tree.summary || {};
      const values = [
        ["Planner", summary.planner],
        ["Nodes", state.nodes.length],
        ["Edges", state.edges.length],
        ["Steps", summary.steps],
        ["Macro", summary.macro_steps],
      ];
      metrics.innerHTML = values.map(([label, value]) => `
        <div class="metric">
          <div class="label">${esc(label)}</div>
          <div class="value">${esc(value ?? "-")}</div>
        </div>
      `).join("");
    }

    function renderTree() {
      const selectedNode = state.nodeById.get(state.selectedId);
      const selectedParent = selectedNode ? selectedNode.parent_id : null;
      const selectedChildren = new Set(selectedNode ? (selectedNode.children || []) : []);

      const edgeMarkup = state.edges.map((edge) => {
        const parent = state.nodeById.get(edge.parent_id);
        const child = state.nodeById.get(edge.child_id);
        if (!parent || !child) return "";
        const selected = edge.child_id === state.selectedId || edge.parent_id === state.selectedId;
        const dim = !parent._match && !child._match;
        const x1 = parent._x + 28;
        const y1 = parent._y;
        const x2 = child._x - 28;
        const y2 = child._y;
        const mid = Math.max(55, (x2 - x1) * 0.55);
        return `<path class="edge ${selected ? "selected" : ""}" opacity="${dim ? 0.18 : 0.82}"
          d="M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}">
          <title>${esc(edge.id)} ${esc(edge.source || "")} reward=${esc(fmt(sum(edge.rewards), 3))}</title>
        </path>`;
      }).join("");

      const nodeMarkup = state.nodes.map((node) => {
        const selected = node.id === state.selectedId;
        const related = node.id === selectedParent || selectedChildren.has(node.id);
        const opacity = node._match || selected || related ? 1 : 0.22;
        const radius = selected ? 17 : node.terminal ? 15 : 13;
        const score = nodeScore(node);
        const color = scoreColor(score, node.success);
        const label = `${node.id.replace("n", "")}  ${fmt(score, 2)}`;
        return `<g class="node ${selected ? "selected" : ""}" opacity="${opacity}"
            transform="translate(${node._x}, ${node._y})" data-node-id="${esc(node.id)}">
          <circle r="${radius}" fill="${color}"></circle>
          <circle class="node-hit" r="28"></circle>
          <text x="24" y="4">${esc(label)}</text>
          <title>${esc(node.id)} score=${esc(fmt(score, 3))} step=${esc(node.env_step)}</title>
        </g>`;
      }).join("");

      svg.setAttribute("viewBox", `0 0 ${svg.clientWidth || 800} ${svg.clientHeight || 600}`);
      svg.innerHTML = `
        <g transform="translate(${state.panX}, ${state.panY}) scale(${state.zoom})">
          <g>${edgeMarkup}</g>
          <g>${nodeMarkup}</g>
        </g>
      `;

    }

    function sum(values) {
      return Array.isArray(values) ? values.reduce((total, value) => total + Number(value || 0), 0) : 0;
    }

    function renderDetails() {
      const node = state.nodeById.get(state.selectedId);
      if (!node) {
        details.innerHTML = `<div class="panel"><div class="empty">No node selected</div></div>`;
        return;
      }
      const incoming = state.edgeByChild.get(node.id);
      const children = (node.children || []).map((id) => state.nodeById.get(id)).filter(Boolean);

      details.innerHTML = `
        <section class="panel">
          <div class="panel-head">
            <span>${esc(node.id)}</span>
            <span class="${statusClass(node.success)}">${node.success ? "success" : node.terminal ? "terminal" : "active"}</span>
          </div>
          ${node.image_path ? `<div class="image-box"><img src="${imageUrl(node.image_path)}" alt="${esc(node.id)}"></div>` : ""}
          <div class="panel-body">
            <div class="kv">
              ${kv("Score", fmt(nodeScore(node), 3))}
              ${kv("Step", node.env_step)}
              ${kv("Depth", node.depth)}
              ${kv("Visits", node.visits)}
              ${kv("Value Sum", fmt(node.value_sum, 3))}
              ${kv("Reward", fmt(node.reward_sum, 3))}
              ${kv("Planner", node.planner)}
              ${kv("Source", node.source || (incoming && incoming.source) || "-")}
            </div>
          </div>
        </section>

        ${adaptiveNoisePanel("Adaptive Noise", adaptiveNoise(node))}

        <section class="panel">
          <div class="panel-head">Reward Model</div>
          ${vlmGallery(node)}
          <div class="panel-body">
            <div class="text-block"><strong>Reason</strong>\n${esc(nodeReason(node))}</div>
          </div>
          ${jsonPanel(nodeMetadata(node))}
          <div class="panel-body" style="border-top:1px solid var(--line)">
            <div class="text-block"><strong>Prompt</strong>\n${esc(node.prompt || "")}</div>
          </div>
          ${jsonPanel(node.vlm_raw_response)}
        </section>

        ${incoming ? edgePanel("Incoming Edge", incoming) : ""}
        ${childrenPanel(children)}
      `;
    }

    function kv(label, value) {
      return `<div><span>${esc(label)}</span><strong>${esc(value ?? "-")}</strong></div>`;
    }

    function jsonPanel(value) {
      if (!value) return "";
      return `
        <div class="panel-body" style="border-top:1px solid var(--line)">
          <pre>${esc(JSON.stringify(value, null, 2))}</pre>
        </div>
      `;
    }

    function adaptiveNoisePanel(title, info) {
      if (!info) return "";
      return `
        <section class="panel">
          <div class="panel-head">${esc(title)}</div>
          <div class="panel-body">
            <div class="kv">
              ${kv("Enabled", info.enabled)}
              ${kv("Base Std", fmt(info.base_noise_std, 4))}
              ${kv("Effective Std", fmt(info.effective_noise_std, 4))}
              ${kv("Scale", fmt(info.noise_scale, 3))}
              ${kv("Time", fmt(info.time_ratio, 3))}
              ${kv("Score", fmt(info.current_score, 3))}
              ${kv("Stagnation", fmt(info.stagnation, 3))}
              ${kv("Oscillation", fmt(info.oscillation, 3))}
              ${kv("Improvement", fmt(info.improvement, 3))}
              ${kv("Slope", fmt(info.slope, 3))}
              ${kv("Damping", fmt(info.high_score_damping, 3))}
            </div>
          </div>
          ${jsonPanel(info)}
        </section>
      `;
    }

    function arrayFromMetadata(value) {
      return Array.isArray(value) ? value.map(String) : [];
    }

    function classifyVlmImage(item, metadata) {
      const label = String(item.label || "");
      const referenceLabels = new Set(arrayFromMetadata(metadata && metadata.vlm_reference_image_labels));
      const stateLabels = new Set(arrayFromMetadata(metadata && metadata.vlm_state_image_labels));
      if (label.startsWith("reference.target.") || referenceLabels.has(label)) return "reference";
      if (stateLabels.has(label) || label === "rendered_overview" || label.startsWith("observation.")) return "state";
      return "other";
    }

    function groupVlmImages(node) {
      const metadata = nodeMetadata(node) || {};
      const groups = { reference: [], state: [], other: [] };
      const images = Array.isArray(nodeImages(node)) ? nodeImages(node) : [];
      for (const item of images) {
        groups[classifyVlmImage(item, metadata)].push(item);
      }
      return groups;
    }

    function vlmGallery(node) {
      const groups = groupVlmImages(node);
      const total = groups.reference.length + groups.state.length + groups.other.length;
      if (!total) {
        const metadata = nodeMetadata(node);
        const labels = metadata && (metadata.vlm_image_labels || metadata.score_image_labels);
        return `
          <div class="panel-body" style="border-bottom:1px solid var(--line)">
            <div class="empty">
              No VLM input images were stored for this node.
              ${Array.isArray(labels) && labels.length ? `<br>Labels: ${esc(labels.join(", "))}` : ""}
            </div>
          </div>
        `;
      }
      return `
        <div class="panel-body" style="border-bottom:1px solid var(--line)">
          <div class="vlm-summary">
            <span class="pill">${total} images</span>
            <span class="pill">${groups.reference.length} reference</span>
            <span class="pill">${groups.state.length} state</span>
            ${groups.other.length ? `<span class="pill">${groups.other.length} other</span>` : ""}
          </div>
          ${vlmGroup("Reference Target Images", groups.reference, "reference")}
          ${vlmGroup("Current State Images", groups.state, "state")}
          ${groups.other.length ? vlmGroup("Other Images", groups.other, "other") : ""}
        </div>
      `;
    }

    function vlmGroup(title, images, kind) {
      if (!images.length) return "";
      return `
        <div class="vlm-section">
          <div class="vlm-section-title">
            <span>${esc(title)}</span>
            <span>${images.length}</span>
          </div>
          <div class="vlm-gallery">
            ${images.map((item, index) => {
              const label = item.label || `${title} ${index + 1}`;
              const path = item.path || "";
              return `
                <div class="vlm-card ${kind}" title="${esc(label)}"
                    data-preview-src="${esc(imageUrl(path))}" data-preview-label="${esc(label)}">
                  ${path ? `<img src="${imageUrl(path)}" alt="${esc(label)}">` : ""}
                  <div>
                    <strong>${esc(kind)}</strong>
                    <span>${esc(label)}</span>
                  </div>
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    }

    function edgePanel(title, edge) {
      const firstAction = Array.isArray(edge.actions) && edge.actions.length ? edge.actions[0] : null;
      const lastAction = Array.isArray(edge.actions) && edge.actions.length ? edge.actions[edge.actions.length - 1] : null;
      const steps = Array.isArray(edge.step_records) ? edge.step_records : [];
      return `
        <section class="panel">
          <div class="panel-head">${esc(title)} <span class="pill">${esc(edge.id)}</span></div>
          <div class="panel-body">
            <div class="kv">
              ${kv("Source", edge.source)}
              ${kv("Candidate", edge.candidate_index)}
              ${kv("Actions", edge.action_count)}
              ${kv("Noise Std", fmt(edge.noise_std, 4))}
              ${kv("Noise Scale", fmt(edge.noise_scale, 3))}
              ${kv("Noise Mode", edge.noise_mode)}
              ${kv("Reward Sum", fmt(sum(edge.rewards), 3))}
            </div>
          </div>
          ${adaptiveNoisePanel("Edge Adaptive Noise", adaptiveNoise(edge))}
          <div class="panel-body" style="border-top:1px solid var(--line)">
            <pre>${esc(JSON.stringify({ first_action: firstAction, last_action: lastAction }, null, 2))}</pre>
          </div>
          ${steps.length ? `<div class="panel-body" style="border-top:1px solid var(--line)">${stepStrip(steps)}</div>` : ""}
        </section>
      `;
    }

    function stepStrip(steps) {
      return `<div class="steps">` + steps.map((step) => `
        <div class="step" title="step ${esc(step.env_step)} reward ${esc(step.reward)}">
          ${step.image_path ? `<img src="${imageUrl(step.image_path)}" alt="step ${esc(step.env_step)}">` : ""}
          <div>${esc(step.env_step)} r=${esc(fmt(step.reward, 2))}</div>
        </div>
      `).join("") + `</div>`;
    }

    function childrenPanel(children) {
      if (!children.length) return "";
      const rows = children.map((child) => {
        const edge = state.edgeByChild.get(child.id) || {};
        const noise = adaptiveNoise(edge) || adaptiveNoise(child);
        return `<tr class="clickable" data-node-id="${esc(child.id)}">
          <td>${esc(child.id)}</td>
          <td>${esc(fmt(nodeScore(child), 3))}</td>
          <td>${esc(edge.source || child.source || "-")}</td>
          <td>${esc(child.env_step)}</td>
          <td>${noise ? esc(fmt(noise.effective_noise_std, 4)) : "-"}</td>
          <td>${noise ? esc(fmt(noise.stagnation, 2)) : "-"}</td>
          <td>${esc(short(nodeReason(child), 56))}</td>
        </tr>`;
      }).join("");
      return `
        <section class="panel">
          <div class="panel-head">Children</div>
          <table>
            <thead><tr><th>Node</th><th>Score</th><th>Source</th><th>Step</th><th>Noise</th><th>Stag.</th><th>Reason</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </section>
      `;
    }

    function selectNode(id) {
      if (!state.nodeById.has(id)) return;
      state.selectedId = id;
      renderTree();
      renderDetails();
    }

    function openLightbox(src, label) {
      if (!src) return;
      lightboxTitle.textContent = label || "VLM input";
      lightboxImage.src = src;
      lightboxImage.alt = label || "VLM input";
      lightbox.classList.add("open");
    }

    function closeLightbox() {
      lightbox.classList.remove("open");
      lightboxImage.removeAttribute("src");
      lightboxImage.alt = "";
    }

    function closestNodeElement(target) {
      return target instanceof Element ? target.closest(".node") : null;
    }

    function fitTree() {
      const box = svg.getBoundingClientRect();
      const bounds = state.layoutBounds;
      const scaleX = (box.width - 80) / Math.max(1, bounds.width);
      const scaleY = (box.height - 80) / Math.max(1, bounds.height);
      state.zoom = Math.max(0.18, Math.min(1.5, Math.min(scaleX, scaleY)));
      state.panX = 40;
      state.panY = 40;
      renderTree();
    }

    searchInput.addEventListener("input", () => {
      state.query = searchInput.value;
      renderTree();
    });

    depthFilter.addEventListener("change", () => {
      state.depth = depthFilter.value;
      renderTree();
    });

    reloadBtn.addEventListener("click", loadTree);
    fitBtn.addEventListener("click", fitTree);

    details.addEventListener("click", (event) => {
      if (!(event.target instanceof Element)) return;
      const preview = event.target.closest("[data-preview-src]");
      if (preview) {
        openLightbox(preview.dataset.previewSrc, preview.dataset.previewLabel);
        return;
      }
      const row = event.target.closest("[data-node-id]");
      if (row) selectNode(row.dataset.nodeId);
    });

    lightboxClose.addEventListener("click", closeLightbox);
    lightbox.addEventListener("click", (event) => {
      if (
        event.target === lightbox
        || (event.target instanceof Element && event.target.classList.contains("lightbox-body"))
      ) closeLightbox();
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeLightbox();
    });

    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.1 : 0.9;
      state.zoom = Math.max(0.12, Math.min(3.5, state.zoom * factor));
      renderTree();
    }, { passive: false });

    svg.addEventListener("click", (event) => {
      const nodeElement = closestNodeElement(event.target);
      if (!nodeElement) return;
      event.stopPropagation();
      selectNode(nodeElement.dataset.nodeId);
    });

    svg.addEventListener("pointerdown", (event) => {
      if (closestNodeElement(event.target)) return;
      state.dragging = true;
      state.dragStart = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
      svg.classList.add("dragging");
      svg.setPointerCapture(event.pointerId);
    });

    svg.addEventListener("pointermove", (event) => {
      if (!state.dragging || !state.dragStart) return;
      state.panX = state.dragStart.panX + event.clientX - state.dragStart.x;
      state.panY = state.dragStart.panY + event.clientY - state.dragStart.y;
      renderTree();
    });

    svg.addEventListener("pointerup", () => {
      state.dragging = false;
      state.dragStart = null;
      svg.classList.remove("dragging");
    });

    window.addEventListener("resize", renderTree);
    loadTree();
  </script>
</body>
</html>
"""


class TraceViewerHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, trace_dir: Path, **kwargs) -> None:
        self.trace_dir = trace_dir
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/tree":
            self._send_tree()
            return
        if parsed.path == "/api/events":
            self._send_events()
            return
        if parsed.path.startswith("/trace/"):
            self._send_trace_file(parsed.path.removeprefix("/trace/"))
            return
        self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send_headers(len(INDEX_HTML.encode("utf-8")), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/tree":
            tree_path = self.trace_dir / "tree.json"
            status = HTTPStatus.OK if tree_path.is_file() else HTTPStatus.NOT_FOUND
            self._send_headers(0, "application/json; charset=utf-8", status=status)
            return
        if parsed.path.startswith("/trace/"):
            relative = Path(unquote(parsed.path.removeprefix("/trace/")))
            target = (self.trace_dir / relative).resolve()
            if not relative.is_absolute() and ".." not in relative.parts and target.is_relative_to(self.trace_dir):
                content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                status = HTTPStatus.OK if target.is_file() else HTTPStatus.NOT_FOUND
                self._send_headers(target.stat().st_size if target.is_file() else 0, content_type, status=status)
                return
        self._send_headers(0, "application/json; charset=utf-8", status=HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _send_tree(self) -> None:
        tree_path = self.trace_dir / "tree.json"
        if not tree_path.is_file():
            self._send_json(
                {"error": f"missing tree.json in {self.trace_dir}"},
                status=HTTPStatus.NOT_FOUND,
            )
            return
        try:
            payload = json.loads(tree_path.read_text())
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"invalid tree.json: {exc}"}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload)

    def _send_events(self) -> None:
        events_path = self.trace_dir / "events.jsonl"
        if not events_path.is_file():
            self._send_json({"events": []})
            return
        events = []
        with events_path.open() as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    events.append({"type": "parse_error", "raw": line})
        self._send_json({"events": events})

    def _send_trace_file(self, raw_path: str) -> None:
        relative = Path(unquote(raw_path))
        if relative.is_absolute() or ".." in relative.parts:
            self._send_json({"error": "invalid path"}, status=HTTPStatus.BAD_REQUEST)
            return

        target = (self.trace_dir / relative).resolve()
        if not target.is_relative_to(self.trace_dir) or not target.is_file():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), content_type)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            status=status,
        )

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send_headers(len(body), content_type, status=status)
        self.wfile.write(body)

    def _send_headers(
        self,
        content_length: int,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a browser viewer for a tree-search trace.")
    parser.add_argument("--trace_dir", type=Path, required=True, help="Directory containing tree.json.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=7860, help="Bind port.")
    parser.add_argument("--open", action="store_true", help="Open the viewer in a browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_dir = args.trace_dir.expanduser().resolve()
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory does not exist: {trace_dir}")
    if not trace_dir.is_dir():
        raise NotADirectoryError(f"Trace path is not a directory: {trace_dir}")

    handler = partial(TraceViewerHandler, trace_dir=trace_dir)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving trace viewer at {url}")
    print(f"Trace directory: {trace_dir}")
    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
