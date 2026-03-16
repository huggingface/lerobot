#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate an interactive eval leaderboard from Hub model repos.

Reads eval results (as pushed by ``lerobot-eval --push_to_hub``) from one or
more Hugging Face model repos and produces a self-contained HTML page with a
sortable, filterable leaderboard table.

Usage::

    lerobot-leaderboard \
        --repo-ids user/model_a,user/model_b \
        --output leaderboard.html

    # Or from a file listing repo IDs (one per line):
    lerobot-leaderboard \
        --repo-ids-file models.txt \
        --output leaderboard.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    repo_id: str
    policy_type: str = "—"
    dataset: str = "—"
    training_steps: str = "—"
    batch_size: str = "—"
    # env_type -> {group_name -> pc_success}
    eval_results: dict[str, dict[str, float]] = field(default_factory=dict)
    # env_type -> overall pc_success
    eval_overall: dict[str, float] = field(default_factory=dict)
    # env_type -> n_episodes
    eval_n_episodes: dict[str, int] = field(default_factory=dict)


def _try_download(repo_id: str, filename: str) -> dict | None:
    """Download a JSON file from a Hub repo, return parsed dict or None."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(repo_id, filename, repo_type="model")
        with open(path) as f:
            return json.load(f)
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        return None


def _list_eval_dirs(repo_id: str) -> list[str]:
    """List env_type subdirectories under eval/ in a Hub repo."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, repo_type="model")
    except RepositoryNotFoundError:
        return []

    env_types = set()
    for f in files:
        if f.startswith("eval/") and f.count("/") >= 2:
            env_types.add(f.split("/")[1])
    return sorted(env_types)


def fetch_model_entry(repo_id: str) -> ModelEntry:
    """Fetch all available metadata and eval results for a single model."""
    entry = ModelEntry(repo_id=repo_id)

    # Policy config
    policy_cfg = _try_download(repo_id, "config.json")
    if policy_cfg:
        entry.policy_type = policy_cfg.get("type", "—")

    # Training config
    train_cfg = _try_download(repo_id, "train_config.json")
    if train_cfg:
        ds = train_cfg.get("dataset", {})
        entry.dataset = ds.get("repo_id", "—") if isinstance(ds, dict) else str(ds)
        entry.training_steps = str(train_cfg.get("steps", "—"))
        entry.batch_size = str(train_cfg.get("batch_size", "—"))

    # Eval results per env_type
    for env_type in _list_eval_dirs(repo_id):
        eval_info = _try_download(repo_id, f"eval/{env_type}/eval_info.json")
        if not eval_info:
            continue

        per_group = eval_info.get("per_group", {})
        group_results = {}
        for group_name, stats in per_group.items():
            group_results[group_name] = stats.get("pc_success", float("nan"))

        entry.eval_results[env_type] = group_results

        overall = eval_info.get("overall", {})
        entry.eval_overall[env_type] = overall.get("pc_success", float("nan"))
        entry.eval_n_episodes[env_type] = overall.get("n_episodes", 0)

    return entry


def fetch_all(repo_ids: list[str]) -> list[ModelEntry]:
    entries = []
    for repo_id in repo_ids:
        logger.info(f"Fetching {repo_id}...")
        try:
            entries.append(fetch_model_entry(repo_id))
        except Exception as e:
            logger.warning(f"Failed to fetch {repo_id}: {e}")
    return entries


def collect_all_env_types(entries: list[ModelEntry]) -> list[str]:
    """Collect all unique env_types across all entries, sorted."""
    env_types: set[str] = set()
    for e in entries:
        env_types.update(e.eval_overall.keys())
    return sorted(env_types)


def collect_all_groups(entries: list[ModelEntry]) -> dict[str, list[str]]:
    """Collect all unique group names per env_type."""
    groups: dict[str, set[str]] = {}
    for e in entries:
        for env_type, group_results in e.eval_results.items():
            groups.setdefault(env_type, set()).update(group_results.keys())
    return {k: sorted(v) for k, v in groups.items()}


def build_html(entries: list[ModelEntry], title: str = "LeRobot Eval Leaderboard") -> str:
    env_types = collect_all_env_types(entries)
    all_groups = collect_all_groups(entries)

    # Build column structure: fixed cols + per env_type (overall + per-group sub-columns)
    # We'll build the data as JSON and let JS handle rendering
    table_data = []
    for e in entries:
        row = {
            "repo_id": e.repo_id,
            "policy_type": e.policy_type,
            "dataset": e.dataset,
            "training_steps": e.training_steps,
            "batch_size": e.batch_size,
        }
        for env_type in env_types:
            overall = e.eval_overall.get(env_type)
            row[f"{env_type}__overall"] = round(overall, 1) if overall is not None else None
            n_ep = e.eval_n_episodes.get(env_type)
            row[f"{env_type}__n_episodes"] = n_ep if n_ep else None
            for group in all_groups.get(env_type, []):
                val = e.eval_results.get(env_type, {}).get(group)
                row[f"{env_type}__{group}"] = round(val, 1) if val is not None else None
        table_data.append(row)

    # Build column definitions for the JS table
    columns_json = json.dumps(_build_column_defs(env_types, all_groups))
    data_json = json.dumps(table_data)

    return _HTML_TEMPLATE.format(
        title=title,
        columns_json=columns_json,
        data_json=data_json,
    )


def _build_column_defs(env_types: list[str], all_groups: dict[str, list[str]]) -> list[dict]:
    cols = [
        {"key": "repo_id", "label": "Model", "group": "Model Info", "sortable": True, "type": "link"},
        {"key": "policy_type", "label": "Policy", "group": "Model Info", "sortable": True, "type": "text"},
        {"key": "dataset", "label": "Dataset", "group": "Model Info", "sortable": True, "type": "text"},
        {
            "key": "training_steps",
            "label": "Steps",
            "group": "Training",
            "sortable": True,
            "type": "number",
        },
        {
            "key": "batch_size",
            "label": "Batch",
            "group": "Training",
            "sortable": True,
            "type": "number",
        },
    ]
    for env_type in env_types:
        cols.append(
            {
                "key": f"{env_type}__overall",
                "label": "Overall %",
                "group": env_type,
                "sortable": True,
                "type": "pct",
            }
        )
        for group in all_groups.get(env_type, []):
            cols.append(
                {
                    "key": f"{env_type}__{group}",
                    "label": f"{group} %",
                    "group": env_type,
                    "sortable": True,
                    "type": "pct",
                }
            )
        cols.append(
            {
                "key": f"{env_type}__n_episodes",
                "label": "Episodes",
                "group": env_type,
                "sortable": True,
                "type": "number",
            }
        )
    return cols


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title}</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --yellow: #d29922;
    --red: #f85149;
    --header-bg: #1c2128;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 24px;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 1.75rem;
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .subtitle {{
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-bottom: 20px;
  }}
  .controls {{
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .controls input {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.875rem;
    width: 280px;
    outline: none;
  }}
  .controls input:focus {{ border-color: var(--accent); }}
  .controls label {{
    color: var(--text-muted);
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
  }}
  .controls input[type="checkbox"] {{
    width: auto;
    accent-color: var(--accent);
  }}
  .table-wrap {{
    overflow-x: auto;
    border: 1px solid var(--border);
    border-radius: 8px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.85rem;
    white-space: nowrap;
  }}
  thead th {{
    background: var(--header-bg);
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
    position: sticky;
    top: 0;
    z-index: 2;
  }}
  thead th:hover {{ color: var(--text); }}
  thead th .arrow {{ margin-left: 4px; opacity: 0.4; }}
  thead th.sorted .arrow {{ opacity: 1; color: var(--accent); }}
  thead tr.group-header th {{
    text-align: center;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    border-right: 1px solid var(--border);
    cursor: default;
  }}
  tbody tr {{ border-bottom: 1px solid var(--border); }}
  tbody tr:hover {{ background: rgba(88,166,255,0.06); }}
  td {{
    padding: 10px 14px;
    vertical-align: middle;
  }}
  td.model-cell a {{
    color: var(--accent);
    text-decoration: none;
    font-weight: 500;
  }}
  td.model-cell a:hover {{ text-decoration: underline; }}
  td.pct {{
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    text-align: right;
  }}
  td.number {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .pct-high {{ color: var(--green); }}
  .pct-mid  {{ color: var(--yellow); }}
  .pct-low  {{ color: var(--red); }}
  .pct-na   {{ color: var(--text-muted); font-weight: 400; }}
  .best-in-col {{ background: rgba(63,185,80,0.12); }}
  footer {{
    margin-top: 20px;
    color: var(--text-muted);
    font-size: 0.75rem;
  }}
  footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>

<h1>&#129302; {title}</h1>
<p class="subtitle">Click any column header to sort. Filter by typing below.</p>

<div class="controls">
  <input type="text" id="filter" placeholder="Filter models..." />
  <label><input type="checkbox" id="toggleGroups" checked /> Show sub-suite columns</label>
</div>

<div class="table-wrap">
  <table id="leaderboard">
    <thead></thead>
    <tbody></tbody>
  </table>
</div>

<footer>
  Generated by <a href="https://github.com/huggingface/lerobot">LeRobot</a> &middot;
  Eval results from <a href="https://huggingface.co">Hugging Face Hub</a>
</footer>

<script>
const COLUMNS = {columns_json};
const DATA    = {data_json};

let sortKey = null, sortAsc = true, showGroups = true;

function pctClass(v) {{
  if (v == null) return 'pct-na';
  if (v >= 70)   return 'pct-high';
  if (v >= 40)   return 'pct-mid';
  return 'pct-low';
}}

function visibleCols() {{
  if (showGroups) return COLUMNS;
  return COLUMNS.filter(c => !c.key.includes('__') || c.key.endsWith('__overall') || c.key.endsWith('__n_episodes'));
}}

function bestPerCol(rows) {{
  const best = {{}};
  for (const c of visibleCols()) {{
    if (c.type !== 'pct') continue;
    let max = -Infinity;
    for (const r of rows) {{
      const v = r[c.key];
      if (v != null && v > max) max = v;
    }}
    best[c.key] = max === -Infinity ? null : max;
  }}
  return best;
}}

function render() {{
  const filter = document.getElementById('filter').value.toLowerCase();
  let rows = DATA.filter(r => r.repo_id.toLowerCase().includes(filter)
    || (r.policy_type||'').toLowerCase().includes(filter)
    || (r.dataset||'').toLowerCase().includes(filter));

  if (sortKey) {{
    rows.sort((a, b) => {{
      let va = a[sortKey], vb = b[sortKey];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string') va = va.toLowerCase();
      if (typeof vb === 'string') vb = vb.toLowerCase();
      return sortAsc ? (va < vb ? -1 : va > vb ? 1 : 0) : (va > vb ? -1 : va < vb ? 1 : 0);
    }});
  }}

  const cols = visibleCols();
  const best = bestPerCol(rows);

  // Group header row
  const groups = [];
  let lastGroup = null;
  for (const c of cols) {{
    if (c.group !== lastGroup) {{
      groups.push({{ label: c.group, span: 1 }});
      lastGroup = c.group;
    }} else {{
      groups[groups.length - 1].span++;
    }}
  }}

  const thead = document.querySelector('#leaderboard thead');
  thead.innerHTML = '';

  // Group header
  const gtr = document.createElement('tr');
  gtr.className = 'group-header';
  for (const g of groups) {{
    const th = document.createElement('th');
    th.colSpan = g.span;
    th.textContent = g.label;
    gtr.appendChild(th);
  }}
  thead.appendChild(gtr);

  // Column headers
  const htr = document.createElement('tr');
  for (const c of cols) {{
    const th = document.createElement('th');
    th.innerHTML = c.label + ' <span class="arrow">' + (sortKey === c.key ? (sortAsc ? '&#9650;' : '&#9660;') : '&#8693;') + '</span>';
    if (sortKey === c.key) th.classList.add('sorted');
    th.addEventListener('click', () => {{
      if (sortKey === c.key) {{ sortAsc = !sortAsc; }}
      else {{ sortKey = c.key; sortAsc = c.type === 'pct' ? false : true; }}
      render();
    }});
    htr.appendChild(th);
  }}
  thead.appendChild(htr);

  // Body
  const tbody = document.querySelector('#leaderboard tbody');
  tbody.innerHTML = '';
  for (const r of rows) {{
    const tr = document.createElement('tr');
    for (const c of cols) {{
      const td = document.createElement('td');
      const v = r[c.key];
      if (c.type === 'link') {{
        td.className = 'model-cell';
        const a = document.createElement('a');
        a.href = 'https://huggingface.co/' + v;
        a.target = '_blank';
        a.textContent = v;
        td.appendChild(a);
      }} else if (c.type === 'pct') {{
        td.className = 'pct ' + pctClass(v);
        td.textContent = v != null ? v.toFixed(1) : '—';
        if (v != null && best[c.key] != null && v === best[c.key] && rows.length > 1) {{
          td.classList.add('best-in-col');
        }}
      }} else if (c.type === 'number') {{
        td.className = 'number';
        td.textContent = v != null ? v : '—';
      }} else {{
        td.textContent = v || '—';
      }}
      tr.appendChild(td);
    }}
    tbody.appendChild(tr);
  }}
}}

document.getElementById('filter').addEventListener('input', render);
document.getElementById('toggleGroups').addEventListener('change', (e) => {{
  showGroups = e.target.checked;
  render();
}});

render();
</script>
</body>
</html>
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an interactive eval leaderboard from Hub model repos.",
    )
    p.add_argument(
        "--repo-ids",
        type=str,
        default=None,
        help="Comma-separated list of HF model repo IDs.",
    )
    p.add_argument(
        "--repo-ids-file",
        type=str,
        default=None,
        help="Path to a text file with one repo ID per line.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="leaderboard.html",
        help="Output HTML file path (default: leaderboard.html).",
    )
    p.add_argument(
        "--title",
        type=str,
        default="LeRobot Eval Leaderboard",
        help="Title shown in the leaderboard page.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    repo_ids: list[str] = []
    if args.repo_ids:
        repo_ids.extend(r.strip() for r in args.repo_ids.split(",") if r.strip())
    if args.repo_ids_file:
        path = Path(args.repo_ids_file)
        if not path.exists():
            logger.error(f"File not found: {path}")
            sys.exit(1)
        repo_ids.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not repo_ids:
        logger.error("No repo IDs provided. Use --repo-ids or --repo-ids-file.")
        sys.exit(1)

    entries = fetch_all(repo_ids)
    if not entries:
        logger.error("No valid entries found.")
        sys.exit(1)

    html = build_html(entries, title=args.title)
    out = Path(args.output)
    out.write_text(html)
    logger.info(f"Leaderboard written to {out.resolve()}")


if __name__ == "__main__":
    main()
