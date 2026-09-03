"""Render the run registry as a git-log-style lineage graph (self-contained HTML).

Runs form a forest: each run points at its ``parent_run_id`` (the run it resumed
from), so a family of resumes is a chain and a re-branch from an old checkpoint is
a fork. This module lays that forest out like ``git log --graph`` — newest at the
top, one lane per family, dots on a timeline — and renders a details panel that
shows each run's parameters with the values that *changed from its parent* marked,
the way a diff highlights what a commit touched.

Pure Python + stdlib (torch-free): it only reads the JSON the Registry already
exposes, so ``kaisparov runs graph`` stays as light as ``runs list``.
"""

from __future__ import annotations

import html
from typing import Any

# --- layout constants (SVG units == pixels) ---------------------------------
ROW = 58  # vertical distance between two runs
LANE = 26  # horizontal distance between two lanes
MARGIN = 20  # padding around the graph
DOT = 7  # node radius

_STATUS_COLOR = {
    "completed": "#3fb950",
    "running": "#58a6ff",
    "failed": "#f85149",
    "interrupted": "#d29922",
}
# Lane colours cycle so neighbouring families stay distinguishable.
_LANE_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba", "#39c5cf", "#ff7b72"]


# --------------------------------------------------------------------------- data
def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _order(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Newest first (top of the graph), matching ``git log``."""
    return sorted(runs, key=lambda r: (r.get("created_at", ""), r.get("run_id", "")), reverse=True)


def _assign_lanes(ordered: list[dict[str, Any]]) -> dict[str, int]:
    """Give every run a lane (column), git-graph style.

    Walking newest -> oldest, each open lane "waits" for a specific ancestor id.
    A run drops into the lane that was waiting for it (continuing that family);
    sibling branches that were all waiting for the same ancestor converge onto it.
    A run nobody was waiting for is a fresh head and opens a new lane.
    """
    ids = {r["run_id"] for r in ordered}
    lanes: list[str | None] = []  # lanes[i] = ancestor id that lane i is waiting for
    node_lane: dict[str, int] = {}

    def _free_lane() -> int:
        for i, waiting in enumerate(lanes):
            if waiting is None:
                return i
        lanes.append(None)
        return len(lanes) - 1

    for run in ordered:
        rid = run["run_id"]
        waiting_for_me = [i for i, w in enumerate(lanes) if w == rid]
        if waiting_for_me:
            lane = min(waiting_for_me)
            for i in waiting_for_me:
                if i != lane:
                    lanes[i] = None  # branch merges upward into this ancestor
        else:
            lane = _free_lane()
        node_lane[rid] = lane

        parent = run.get("parent_run_id")
        # Keep waiting only for a parent we can actually draw; otherwise close the lane.
        lanes[lane] = parent if (parent in ids) else None

    return node_lane


def _lane_x(lane: int) -> int:
    return MARGIN + lane * LANE


def _latest_eval(run: dict[str, Any]) -> str:
    history = run.get("eval_history") or []
    if not history:
        return ""
    last = history[-1]
    for key in ("elo_vs_random", "winrate_vs_random", "elo_vs_material"):
        if key in last:
            label = key.replace("_vs_", " vs ").replace("elo", "Elo").replace("winrate", "win%")
            val = last[key] * 100 if key.startswith("winrate") else last[key]
            unit = "%" if key.startswith("winrate") else ""
            return f"{label} {val:.0f}{unit}"
    return ""


# --------------------------------------------------------------------------- SVG
def _render_graph(ordered: list[dict[str, Any]], node_lane: dict[str, int]) -> str:
    row_of = {r["run_id"]: i for i, r in enumerate(ordered)}
    max_lane = max(node_lane.values(), default=0)
    width = MARGIN * 2 + max_lane * LANE
    height = max(len(ordered) * ROW, ROW)

    def _cx(rid: str) -> int:
        return _lane_x(node_lane[rid])

    def _cy(rid: str) -> int:
        return row_of[rid] * ROW + ROW // 2

    edges, dots = [], []
    for run in ordered:
        rid = run["run_id"]
        x, y = _cx(rid), _cy(rid)
        color = _LANE_COLORS[node_lane[rid] % len(_LANE_COLORS)]
        parent = run.get("parent_run_id")
        if parent in row_of:  # draw a link down to the ancestor
            px, py = _cx(parent), _cy(parent)
            if px == x:
                path = f"M{x},{y} L{px},{py}"
            else:  # curve across lanes near the ancestor's row (branch/merge)
                mid = py - ROW // 2
                path = f"M{x},{y} L{x},{mid} C{x},{py} {px},{mid} {px},{py}"
            edges.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')

        status = run.get("status", "")
        fill = _STATUS_COLOR.get(status, "#8b949e")
        dots.append(
            f'<circle cx="{x}" cy="{y}" r="{DOT}" fill="{fill}" stroke="var(--bg)" '
            f'stroke-width="2"><title>{_esc(rid)} ({_esc(status)})</title></circle>'
        )

    return (
        f'<svg class="graph" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{"".join(edges)}{"".join(dots)}</svg>'
    )


# ------------------------------------------------------------------ params/diff
def _flatten(cfg: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, full + "."))
        else:
            flat[full] = value
    return flat


def _fmt_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(_fmt_value(v) for v in value)
    return _esc(value)


def _render_params(run: dict[str, Any], parent: dict[str, Any] | None) -> str:
    flat = _flatten(run.get("config") or {})
    parent_flat = _flatten(parent.get("config") or {}) if parent else {}
    rows = []
    for key in sorted(flat):
        value = flat[key]
        changed = parent is not None and parent_flat.get(key) != value
        was = ""
        if changed and key in parent_flat:
            was = f'<span class="was">was {_fmt_value(parent_flat[key])}</span>'
        elif changed and parent is not None:
            was = '<span class="was">new</span>'
        cls = ' class="changed"' if changed else ""
        rows.append(
            f"<tr{cls}><td class='k'>{_esc(key)}</td>"
            f"<td class='v'>{_fmt_value(value)} {was}</td></tr>"
        )
    note = ""
    if parent is not None:
        n_changed = sum(1 for k, v in flat.items() if parent_flat.get(k) != v)
        note = f'<p class="hint">{n_changed} parameter(s) changed vs parent.</p>'
    elif run.get("parent_run_id"):
        note = '<p class="hint">Parent run not in this registry — showing all params.</p>'
    return f'{note}<table class="params">{"".join(rows)}</table>'


def _render_eval(run: dict[str, Any]) -> str:
    history = run.get("eval_history") or []
    if not history:
        return ""
    cols = [k for k in history[-1] if k != "epoch"]
    head = "".join(f"<th>{_esc(c)}</th>" for c in cols)
    body = []
    for row in history:
        cells = "".join(
            f"<td>{row[c]:.2f}</td>" if isinstance(row.get(c), (int, float)) else "<td>-</td>"
            for c in cols
        )
        body.append(f"<tr><td class='k'>epoch {row.get('epoch', '?')}</td>{cells}</tr>")
    return (
        '<h4>Eval history</h4><table class="params"><tr><th></th>'
        f"{head}</tr>{''.join(body)}</table>"
    )


def _render_detail(run: dict[str, Any], parent: dict[str, Any] | None) -> str:
    rid = run["run_id"]
    git = run.get("git") or {}
    commit = git.get("commit") or "?"
    dirty = " (dirty)" if git.get("dirty") else ""
    best = run.get("best_checkpoint")
    best_line = (
        f"<div class='meta'><span>best</span> epoch {best['epoch']} "
        f"({_esc(best.get('metric'))}={best['value']:.2f})</div>"
        if best
        else ""
    )
    parent_line = ""
    if run.get("parent_run_id"):
        parent_line = (
            f"<div class='meta'><span>parent</span> <code>{_esc(run['parent_run_id'])}</code></div>"
        )
    desc = run.get("description") or ""
    desc_html = f"<p class='desc'>{_esc(desc)}</p>" if desc.strip() else ""

    return f"""<div class="detail" id="d-{_esc(rid)}" hidden>
  <h3>{_esc(run.get("title") or rid)}</h3>
  <div class="subid"><code>{_esc(rid)}</code></div>
  {desc_html}
  <div class="metagrid">
    <div class="meta"><span>status</span> {_esc(run.get("status"))}</div>
    <div class="meta"><span>model</span> {_esc(run.get("model"))}</div>
    <div class="meta"><span>epochs</span> {_esc(run.get("epochs_completed", 0))}</div>
    <div class="meta"><span>params</span> {_esc(run.get("num_params"))}</div>
    <div class="meta"><span>seed</span> {_esc(run.get("seed"))}</div>
    <div class="meta"><span>device</span> {_esc(run.get("device"))}</div>
    <div class="meta"><span>commit</span> <code>{_esc(commit)}{dirty}</code></div>
    <div class="meta"><span>created</span> {_esc((run.get("created_at") or "")[:19])}</div>
    {parent_line}
    {best_line}
  </div>
  <h4>Parameters</h4>
  {_render_params(run, parent)}
  {_render_eval(run)}
</div>"""


# --------------------------------------------------------------------------- page
def _render_rows(ordered: list[dict[str, Any]], node_lane: dict[str, int]) -> str:
    rows = []
    for run in ordered:
        rid = run["run_id"]
        color = _LANE_COLORS[node_lane[rid] % len(_LANE_COLORS)]
        status = run.get("status", "")
        pill = _STATUS_COLOR.get(status, "#8b949e")
        title = run.get("title") or rid
        eval_txt = _latest_eval(run)
        eval_html = f"<span class='eval'>{_esc(eval_txt)}</span>" if eval_txt else ""
        rows.append(
            f"""<div class="row" data-id="{_esc(rid)}" onclick="selectRun('{_esc(rid)}')"
     style="height:{ROW}px;--lane:{color}">
  <div class="rowmain">
    <span class="title">{_esc(title)}</span>
    <span class="pill" style="background:{pill}">{_esc(status)}</span>
  </div>
  <div class="rowsub">
    <code>{_esc(rid)}</code>
    <span class="ep">{_esc(run.get("epochs_completed", 0))} ep</span>
    {eval_html}
  </div>
</div>"""
        )
    return "".join(rows)


def build_html(runs: list[dict[str, Any]], *, root: str = "runs") -> str:
    """Return a self-contained HTML page visualising the run lineage."""
    ordered = _order(runs)
    by_id = {r["run_id"]: r for r in runs}
    if not ordered:
        body = "<p class='empty'>No runs found. Train something first.</p>"
        graph = rows = details = ""
    else:
        node_lane = _assign_lanes(ordered)
        graph = _render_graph(ordered, node_lane)
        rows = _render_rows(ordered, node_lane)
        details = "".join(_render_detail(r, by_id.get(r.get("parent_run_id"))) for r in ordered)
        first = ordered[0]["run_id"]
        body = (
            f'<div class="log"><div class="graphcol">{graph}</div>'
            f'<div class="rowscol">{rows}</div></div>'
            f'<aside class="panel" id="panel">{details}'
            f'<p class="placeholder">Select a run to see its parameters.</p></aside>'
            f"<script>const FIRST={first!r};</script>"
        )

    return _PAGE.format(root=_esc(root), count=len(ordered), body=body)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kAIsparov — run lineage</title>
<style>
  :root {{
    --bg:#ffffff; --fg:#1f2328; --muted:#6e7781; --border:#d0d7de;
    --panel:#f6f8fa; --card:#ffffff; --accent:#0969da; --chg:#fff8c5; --chgbd:#d4a72c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --border:#30363d;
      --panel:#161b22; --card:#0d1117; --accent:#58a6ff; --chg:#3b2e12; --chgbd:#9e6a03;
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ padding:14px 20px; border-bottom:1px solid var(--border);
    display:flex; align-items:baseline; gap:12px; position:sticky; top:0; background:var(--bg); z-index:2;}}
  header h1 {{ font-size:16px; margin:0; }}
  header .sub {{ color:var(--muted); font-size:13px; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
  .wrap {{ display:grid; grid-template-columns:minmax(0,1fr) 420px; gap:0; align-items:start; }}
  .log {{ display:flex; padding:12px 0; overflow-x:auto; }}
  .graphcol {{ flex:0 0 auto; }}
  .rowscol {{ flex:1 1 auto; min-width:0; }}
  .row {{ display:flex; flex-direction:column; justify-content:center; padding:0 16px;
    border-left:3px solid transparent; cursor:pointer; }}
  .row:hover {{ background:var(--panel); }}
  .row.active {{ background:var(--panel); border-left-color:var(--lane); }}
  .rowmain {{ display:flex; align-items:center; gap:8px; }}
  .title {{ font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .pill {{ color:#fff; font-size:11px; padding:1px 7px; border-radius:10px; flex:0 0 auto; }}
  .rowsub {{ color:var(--muted); font-size:12px; display:flex; gap:12px; align-items:center; }}
  .rowsub .eval {{ color:var(--accent); }}
  .panel {{ position:sticky; top:52px; height:calc(100vh - 52px); overflow-y:auto;
    border-left:1px solid var(--border); background:var(--panel); padding:18px 20px; }}
  .panel h3 {{ margin:0 0 2px; font-size:15px; }}
  .subid {{ color:var(--muted); margin-bottom:10px; }}
  .desc {{ white-space:pre-wrap; color:var(--fg); background:var(--card);
    border:1px solid var(--border); border-radius:6px; padding:8px 10px; font-size:13px; }}
  .metagrid {{ display:grid; grid-template-columns:1fr 1fr; gap:4px 14px; margin:12px 0; }}
  .meta {{ font-size:13px; }}
  .meta span {{ color:var(--muted); display:inline-block; min-width:58px; }}
  h4 {{ margin:16px 0 6px; font-size:13px; text-transform:uppercase; letter-spacing:.04em;
    color:var(--muted); }}
  table.params {{ width:100%; border-collapse:collapse; font-size:12.5px; }}
  table.params td, table.params th {{ padding:3px 6px; border-bottom:1px solid var(--border);
    text-align:left; vertical-align:top; }}
  table.params th {{ color:var(--muted); font-weight:600; }}
  td.k {{ color:var(--muted); white-space:nowrap; }}
  td.v {{ font-family:ui-monospace,Menlo,monospace; }}
  tr.changed td.v {{ background:var(--chg); border-left:2px solid var(--chgbd); }}
  .was {{ color:var(--muted); font-family:inherit; font-size:11px; }}
  .hint {{ color:var(--muted); font-size:12px; margin:2px 0 8px; }}
  .placeholder {{ color:var(--muted); }}
  .empty {{ padding:40px; color:var(--muted); }}
  @media (max-width:820px) {{
    .wrap {{ grid-template-columns:1fr; }}
    .panel {{ position:static; height:auto; border-left:none; border-top:1px solid var(--border); }}
  }}
</style>
</head>
<body>
<header>
  <h1>kAIsparov — run lineage</h1>
  <span class="sub">{count} run(s) in <code>{root}/</code> · newest first · resume chains linked</span>
</header>
<div class="wrap">{body}</div>
<script>
  function selectRun(id) {{
    document.querySelectorAll('.detail').forEach(d => d.hidden = true);
    document.querySelectorAll('.row').forEach(r => r.classList.remove('active'));
    const d = document.getElementById('d-' + id);
    if (d) d.hidden = false;
    const r = document.querySelector('.row[data-id="' + CSS.escape(id) + '"]');
    if (r) r.classList.add('active');
    const ph = document.querySelector('.placeholder');
    if (ph) ph.style.display = 'none';
  }}
  if (typeof FIRST !== 'undefined') selectRun(FIRST);
</script>
</body>
</html>"""
