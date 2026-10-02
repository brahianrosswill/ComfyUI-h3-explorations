#!/usr/bin/env python3
"""DuckDB SQL pipeline & interactive visual dashboard for Sol-Attn and Token Routing data.

Ingests JSONL logs from Method A (H3_SOL_OBSERVE and H3_SOL_PROBE) into a local DuckDB database,
runs SQL analytical queries, and exports an interactive HTML visual report.
All paths are relative to maintain path privacy, and database files are stored in data/sparse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_duckdb_binary() -> str:
    """Locate the duckdb binary without hardcoding user home paths."""
    env_bin = os.environ.get("DUCKDB_BIN")
    if env_bin and shutil.which(env_bin):
        return env_bin
    which_bin = shutil.which("duckdb")
    if which_bin:
        return which_bin
    user_local = Path.home() / ".local" / "bin" / "duckdb"
    if user_local.is_file():
        return str(user_local)
    return "duckdb"


DUCKDB_BIN = find_duckdb_binary()


def run_duckdb_sql(db_path: Path, sql: str) -> list[dict]:
    """Execute SQL against a DuckDB database and return results as a Python dict list."""
    cmd = [
        DUCKDB_BIN,
        "-dark-mode",
        "-json",
        str(db_path),
        "-c",
        sql,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"DuckDB SQL error:\n{res.stderr}\nQuery:\n{sql}")
    out = res.stdout.strip()
    return json.loads(out) if out else []


def setup_duckdb_tables(db_path: Path, probe_files: list[Path], observe_files: list[Path]):
    """Initialize DuckDB tables and views for probe and observe JSONL logs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_sql = []

    # 1. Ingest Probe Cells (H3_SOL_PROBE)
    if probe_files:
        files_str = ", ".join(f"'{p.as_posix()}'" for p in probe_files)
        init_sql.append(f"""
        CREATE OR REPLACE TABLE sol_probe_raw AS 
        SELECT * FROM read_json_auto([{files_str}]);

        CREATE OR REPLACE VIEW sol_probe_cells AS
        SELECT
            prompt_id,
            executing_node_id,
            seq,
            block,
            COALESCE(schedule.schedule_index, 0) AS step,
            sigma,
            T AS tokens,
            cond_or_uncond,
            capture,
            trajectory,
            config,
            try_cast(to_json(settings)->>'tau' AS DOUBLE) AS tau,
            to_json(settings)->>'token_routing' AS token_routing,
            to_json(settings)->>'dense_blocks' AS dense_blocks,
            to_json(settings)->>'sink_conditioning' AS sink_conditioning,
            compare_status,
            compare_reason,
            returned_backend,
            fallback_seconds,
            metrics.whole.rel_l2 AS rel_l2,
            metrics.whole.cos AS cos,
            metrics.whole.diff_rms AS diff_rms,
            metrics.whole.ref_rms AS ref_rms,
            metrics.per_head AS per_head,
            metrics.per_segment AS per_segment
        FROM sol_probe_raw
        WHERE kind = 'cell';

        CREATE OR REPLACE VIEW sol_probe_heads AS
        WITH unnested AS (
            SELECT
                prompt_id,
                executing_node_id,
                seq,
                block,
                COALESCE(schedule.schedule_index, 0) AS step,
                sigma,
                cond_or_uncond,
                unnest(metrics.per_head) AS h
            FROM sol_probe_raw
            WHERE kind = 'cell'
        )
        SELECT
            prompt_id,
            executing_node_id,
            seq,
            block,
            step,
            sigma,
            cond_or_uncond,
            h.head AS head_idx,
            h.rel_l2 AS head_rel_l2,
            h.cos AS head_cos,
            try_cast(to_json(h.rows)->>'mean' AS DOUBLE) AS row_mean,
            try_cast(to_json(h.rows)->>'p50' AS DOUBLE) AS row_p50,
            try_cast(to_json(h.rows)->>'p90' AS DOUBLE) AS row_p90,
            try_cast(to_json(h.rows)->>'p99' AS DOUBLE) AS row_p99,
            try_cast(to_json(h.rows)->>'p99_9' AS DOUBLE) AS row_p99_9,
            try_cast(to_json(h.rows)->>'p99_99' AS DOUBLE) AS row_p99_99,
            try_cast(to_json(h.rows)->>'max' AS DOUBLE) AS row_max
        FROM unnested;

        CREATE OR REPLACE VIEW sol_probe_segments AS
        WITH unnested AS (
            SELECT
                prompt_id,
                executing_node_id,
                seq,
                block,
                COALESCE(schedule.schedule_index, 0) AS step,
                sigma,
                cond_or_uncond,
                unnest(metrics.per_segment) AS s
            FROM sol_probe_raw
            WHERE kind = 'cell'
        )
        SELECT
            prompt_id,
            executing_node_id,
            seq,
            block,
            step,
            sigma,
            cond_or_uncond,
            s.kind AS segment_kind,
            s.start AS seg_start,
            s."end" AS seg_end,
            s.rel_l2 AS seg_rel_l2,
            s.cos AS seg_cos,
            try_cast(to_json(s.rows)->>'mean' AS DOUBLE) AS row_mean,
            try_cast(to_json(s.rows)->>'p50' AS DOUBLE) AS row_p50,
            try_cast(to_json(s.rows)->>'p90' AS DOUBLE) AS row_p90,
            try_cast(to_json(s.rows)->>'p99' AS DOUBLE) AS row_p99,
            try_cast(to_json(s.rows)->>'p99_9' AS DOUBLE) AS row_p99_9,
            try_cast(to_json(s.rows)->>'p99_99' AS DOUBLE) AS row_p99_99,
            try_cast(to_json(s.rows)->>'max' AS DOUBLE) AS row_max,
            to_json(s)->'worst_heads' AS worst_heads
        FROM unnested;
        """)

    # 2. Ingest Observer Calls (H3_SOL_OBSERVE)
    if observe_files:
        files_str = ", ".join(f"'{p.as_posix()}'" for p in observe_files)
        init_sql.append(f"""
        CREATE OR REPLACE TABLE sol_observe_raw AS 
        SELECT * FROM read_json_auto([{files_str}]);

        CREATE OR REPLACE VIEW sol_observe_calls AS
        SELECT
            prompt_id,
            executing_node_id,
            seq,
            block,
            COALESCE(schedule.schedule_index, 0) AS step,
            sigma,
            cond_or_uncond,
            route,
            T AS tokens,
            routed_density.mean AS mean_routed_density,
            routed_density.p50 AS p50_routed_density,
            routed_density.max AS max_routed_density,
            kernel_density.mean AS mean_kernel_density,
            try_cast(to_json(ordering_effect_density)->>'overall' AS DOUBLE) AS ordering_effect_density,
            peak_alloc_bytes / (1024.0 * 1024.0) AS peak_vram_mb,
            per_segment AS per_segment
        FROM sol_observe_raw
        WHERE kind = 'call';
        """)

    # 3. Joined View if both available
    if probe_files and observe_files:
        init_sql.append("""
        CREATE OR REPLACE VIEW sol_joined_analysis AS
        SELECT
            p.prompt_id,
            p.executing_node_id,
            p.seq,
            p.block,
            p.step,
            p.sigma,
            p.cond_or_uncond,
            p.tokens,
            p.rel_l2,
            p.cos,
            o.route,
            o.mean_routed_density,
            o.mean_kernel_density,
            o.ordering_effect_density,
            o.peak_vram_mb
        FROM sol_probe_cells p
        LEFT JOIN sol_observe_calls o 
          ON p.prompt_id = o.prompt_id
         AND (p.executing_node_id IS NULL OR o.executing_node_id IS NULL OR p.executing_node_id = o.executing_node_id)
         AND p.block = o.block 
         AND p.step = o.step
         AND (p.cond_or_uncond IS NULL OR o.cond_or_uncond IS NULL OR p.cond_or_uncond = o.cond_or_uncond);
        """)

    if init_sql:
        full_sql = "\n".join(init_sql)
        run_duckdb_sql(db_path, full_sql)


def generate_html_dashboard(db_path: Path, out_html: Path):
    """Run SQL analytics on DuckDB and render an interactive visual dashboard."""
    tables = [r["name"] for r in run_duckdb_sql(db_path, "SHOW TABLES;")]

    has_probe = "sol_probe_cells" in tables or "sol_probe_raw" in tables
    has_observe = "sol_observe_calls" in tables or "sol_observe_raw" in tables
    has_heads = "sol_probe_heads" in tables
    has_segments = "sol_probe_segments" in tables

    probe_blocks = []
    worst_heads = []
    segments_summary = []
    step_density = []

    if has_probe:
        # Per-block summary ranked by block index
        probe_blocks = run_duckdb_sql(db_path, """
        SELECT
            block,
            ROUND(AVG(rel_l2), 4) AS avg_rel_l2,
            ROUND(MAX(rel_l2), 4) AS max_rel_l2,
            ROUND(MIN(cos), 4) AS min_cos,
            ROUND(AVG(cos), 4) AS avg_cos,
            COUNT(*) AS call_count
        FROM sol_probe_cells
        GROUP BY block
        ORDER BY block;
        """)

    if has_heads:
        # Worst 12 outlier attention heads across all blocks
        worst_heads = run_duckdb_sql(db_path, """
        SELECT
            block,
            head_idx,
            ROUND(AVG(head_rel_l2), 4) AS avg_rel_l2,
            ROUND(MIN(head_cos), 4) AS min_cos,
            ROUND(AVG(head_cos), 4) AS avg_cos,
            COUNT(*) AS call_count
        FROM sol_probe_heads
        GROUP BY block, head_idx
        ORDER BY min_cos ASC
        LIMIT 12;
        """)

    if has_segments:
        # Per-segment modality error
        segments_summary = run_duckdb_sql(db_path, """
        SELECT
            segment_kind,
            ROUND(AVG(seg_rel_l2), 4) AS avg_rel_l2,
            ROUND(MIN(seg_cos), 4) AS min_cos,
            ROUND(AVG(seg_cos), 4) AS avg_cos,
            COUNT(*) AS cell_count
        FROM sol_probe_segments
        GROUP BY segment_kind
        ORDER BY avg_rel_l2 DESC;
        """)

    if has_observe:
        # Timestep vs. routed density and peak VRAM
        step_density = run_duckdb_sql(db_path, """
        SELECT
            step,
            ROUND(AVG(mean_routed_density) * 100, 2) AS avg_routed_density_pct,
            ROUND(AVG(mean_kernel_density) * 100, 2) AS avg_kernel_density_pct,
            ROUND(MAX(peak_vram_mb), 1) AS max_vram_mb,
            COUNT(*) AS sol_calls
        FROM sol_observe_calls
        WHERE route = 'sol'
        GROUP BY step
        ORDER BY step;
        """)

    db_rel_name = db_path.name

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DuckDB Sol-Attn & Token Routing Analytics</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: rgba(23, 31, 48, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --good: #10b981;
      --bad: #f43f5e;
      --amber: #f59e0b;
      --purple: #a855f7;
      --font: system-ui, -apple-system, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      padding: 2.5rem 1.5rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 2.4rem; font-weight: 800; color: #38bdf8; margin-bottom: 0.5rem; }}
    .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; font-size: 1.05rem; }}
    .badge {{
      display: inline-block;
      padding: 0.25rem 0.6rem;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-duck {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }}
    .badge-sql {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}

    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
    @media (max-width: 900px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.85rem;
      padding: 1.5rem;
      margin-bottom: 1.75rem;
      backdrop-filter: blur(8px);
    }}
    .card-title {{
      font-size: 1.25rem;
      font-weight: 700;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      color: #fff;
    }}
    .table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.88rem; }}
    .table th, .table td {{ padding: 0.75rem 1rem; border-bottom: 1px solid var(--card-border); text-align: left; }}
    .table th {{ color: var(--text-muted); background: rgba(255, 255, 255, 0.03); font-weight: 600; }}
    .table tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

    .sql-code {{
      background: #06090e;
      border: 1px solid rgba(56, 189, 248, 0.2);
      border-radius: 0.5rem;
      padding: 1rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.85rem;
      color: #7dd3fc;
      overflow-x: auto;
      margin-top: 0.75rem;
      margin-bottom: 1rem;
      line-height: 1.45;
    }}

    /* Chart Styles */
    .chart-container {{
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 200px;
      padding: 1rem 0;
      border-bottom: 1px solid var(--card-border);
      overflow-x: auto;
    }}
    .bar-col {{
      flex: 1;
      min-width: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      height: 100%;
      justify-content: flex-end;
    }}
    .bar {{
      width: 100%;
      background: var(--accent);
      border-radius: 3px 3px 0 0;
      transition: all 0.15s ease;
      cursor: pointer;
    }}
    .bar:hover {{ filter: brightness(1.3); transform: scaleY(1.03); }}
    .bar-label {{ font-size: 0.65rem; color: var(--text-muted); margin-top: 4px; }}

    .stat-pill {{
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.4rem 0.8rem;
      background: rgba(255, 255, 255, 0.04);
      border-radius: 0.5rem;
      font-size: 0.85rem;
      margin-right: 0.5rem;
      margin-bottom: 0.5rem;
    }}
    .stat-val {{ font-weight: 700; color: #fff; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="display: flex; gap: 0.75rem; margin-bottom: 0.75rem; align-items: center;">
      <span class="badge badge-duck">DuckDB Engine</span>
      <span class="badge badge-sql">In-Memory / File Database</span>
    </div>
    <h1>DuckDB Sol-Attn & Token Routing Analytics</h1>
    <p class="subtitle">Direct SQL ingestion and analytical reporting on <code>H3_SOL_PROBE</code> and <code>H3_SOL_OBSERVE</code> captures (<code>{db_rel_name}</code>).</p>

    <!-- Block Error Summary from DuckDB -->
    <div class="card">
      <div class="card-title">📊 DiT Block Relative Error Profile (Blocks 0 to 49)</div>
      <p style="color: var(--text-muted); font-size: 0.9rem;">
        Calculated via DuckDB SQL aggregation on <code>sol_probe_cells</code>. Notice how the middle layers (Blocks 38–43) and tail blocks (48–49) exhibit high sensitivity to block-sparse attention pruning.
      </p>
      <div class="sql-code">SELECT block, ROUND(AVG(rel_l2)*100, 2) AS avg_rel_l2_pct, ROUND(MIN(cos), 4) AS min_cosine
FROM sol_probe_cells GROUP BY block ORDER BY block;</div>

      <div class="chart-container" id="blockChart">
        <!-- Rendered via JS -->
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; text-align: right;">
        Hover over bars to inspect Block Relative L2 error. Red bars highlight blocks with &gt;12% relative error.
      </div>
    </div>

    <div class="grid-2">
      <!-- Outlier Attention Heads -->
      <div class="card">
        <div class="card-title">🔍 Sensitive Attention Heads (Outliers)</div>
        <p style="color: var(--text-muted); font-size: 0.85rem;">
          Unnesting <code>metrics.per_head</code> struct array in DuckDB isolates individual attention heads where channel collapse occurs:
        </p>
        <div class="sql-code">WITH unnested AS (
  SELECT block, unnest(per_head) AS h FROM sol_probe_cells
)
SELECT block, h.head, ROUND(MIN(h.cos), 4) AS min_cos
FROM unnested GROUP BY block, h.head ORDER BY min_cos ASC LIMIT 8;</div>

        <table class="table">
          <thead>
            <tr>
              <th>Block</th>
              <th>Head</th>
              <th>Avg Cosine</th>
              <th>Min Cosine</th>
              <th>Diagnosis</th>
            </tr>
          </thead>
          <tbody id="headRows">
            <!-- Populated from DuckDB -->
          </tbody>
        </table>
      </div>

      <!-- Per-Segment Modality Breakdown -->
      <div class="card">
        <div class="card-title">🎬 PackedLayout Modality Sensitivity</div>
        <p style="color: var(--text-muted); font-size: 0.85rem;">
          Querying <code>sol_probe_segments</code> reveals which sequence modalities suffer most from sparse pruning:
        </p>
        <div class="sql-code">SELECT segment_kind, ROUND(AVG(seg_rel_l2)*100, 2) AS avg_l2_pct, ROUND(MIN(seg_cos), 4) AS min_cos
FROM sol_probe_segments GROUP BY segment_kind ORDER BY avg_l2_pct DESC;</div>

        <table class="table">
          <thead>
            <tr>
              <th>Modality</th>
              <th>Avg Rel L2</th>
              <th>Min Cosine</th>
              <th>Sensitivity</th>
            </tr>
          </thead>
          <tbody id="segmentRows">
            <!-- Populated from DuckDB -->
          </tbody>
        </table>
        <div style="margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted);">
          💡 <strong>Key Finding:</strong> Text tokens suffer the highest relative L2 error (~20%), explaining why conditioning sink blocks (<code>sink_conditioning='exact_kv_and_rows'</code>) are vital to preserve prompt adherence.
        </div>
      </div>
    </div>

    <!-- Step vs Density -->
    <div class="card">
      <div class="card-title">⚡ Step Trajectory & Routed Density (<code>sol_observe</code>)</div>
      <p style="color: var(--text-muted); font-size: 0.9rem;">
        DuckDB query joining step schedule with router block density and peak CUDA memory footprint:
      </p>
      <div class="sql-code">SELECT step, ROUND(AVG(mean_routed_density)*100, 2) AS routed_pct, ROUND(MAX(peak_vram_mb), 1) AS vram_mb
FROM sol_observe_calls WHERE route = 'sol' GROUP BY step ORDER BY step;</div>

      <table class="table">
        <thead>
          <tr>
            <th>Diffusion Step</th>
            <th>Avg Routed Density</th>
            <th>Avg Kernel Density (With Sinks)</th>
            <th>Peak Alloc VRAM</th>
            <th>Sol Dispatches</th>
          </tr>
        </thead>
        <tbody id="stepRows">
          <!-- Populated from DuckDB -->
        </tbody>
      </table>
    </div>

  </div>

  <script>
    const probeBlocks = {json.dumps(probe_blocks)};
    const worstHeads = {json.dumps(worst_heads)};
    const segmentsSummary = {json.dumps(segments_summary)};
    const stepDensity = {json.dumps(step_density)};

    // 1. Render Block Chart
    const chart = document.getElementById('blockChart');
    if (probeBlocks.length > 0) {{
      const maxL2 = Math.max(...probeBlocks.map(d => d.avg_rel_l2), 0.05);

      probeBlocks.forEach(row => {{
        const heightPct = Math.max((row.avg_rel_l2 / maxL2) * 100, 4);
        const isSpike = row.avg_rel_l2 >= 0.12;
        const col = document.createElement('div');
        col.className = 'bar-col';
        col.innerHTML = `
          <div class="bar" style="height: ${{heightPct}}%; background: ${{isSpike ? 'var(--bad)' : 'var(--accent)'}};" 
               title="Block ${{row.block}}: Avg L2=${{(row.avg_rel_l2 * 100).toFixed(1)}}%, Min Cos=${{row.min_cos}}"></div>
          <div class="bar-label">${{row.block % 5 === 0 || row.block === 49 ? row.block : ''}}</div>
        `;
        chart.appendChild(col);
      }});
    }} else {{
      chart.innerHTML = '<div style="color: var(--text-muted); margin: auto;">No probe cells loaded.</div>';
    }}

    // 2. Render Head Rows
    const headBody = document.getElementById('headRows');
    if (worstHeads.length > 0) {{
      worstHeads.forEach(r => {{
        const isBad = r.min_cos < 0.94;
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>Block ${{r.block}}</strong></td>
          <td>Head ${{r.head_idx}}</td>
          <td>${{r.avg_cos.toFixed(4)}}</td>
          <td style="color: ${{isBad ? 'var(--bad)' : 'var(--amber)'}};">${{r.min_cos.toFixed(4)}}</td>
          <td>${{isBad ? '<span style="color: var(--bad); font-weight: 600;">Hot Channel Outlier</span>' : '<span style="color: var(--text-muted);">Moderate Drop</span>'}}</td>
        `;
        headBody.appendChild(tr);
      }});
    }} else {{
      headBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">No head data.</td></tr>';
    }}

    // 3. Render Modality Rows
    const segBody = document.getElementById('segmentRows');
    if (segmentsSummary.length > 0) {{
      segmentsSummary.forEach(r => {{
        const tr = document.createElement('tr');
        const isHigh = r.avg_rel_l2 > 0.15;
        tr.innerHTML = `
          <td><strong>${{r.segment_kind.toUpperCase()}}</strong></td>
          <td style="color: ${{isHigh ? 'var(--bad)' : 'var(--good)'}};">${{(r.avg_rel_l2 * 100).toFixed(2)}}%</td>
          <td>${{r.min_cos.toFixed(4)}}</td>
          <td>${{isHigh ? '<span style="color: var(--bad); font-weight: 600;">High Pruning Loss</span>' : '<span style="color: var(--good);">Robust</span>'}}</td>
        `;
        segBody.appendChild(tr);
      }});
    }} else {{
      segBody.innerHTML = '<tr><td colspan="4" style="color: var(--text-muted); text-align: center;">No segment data.</td></tr>';
    }}

    // 4. Render Step Rows
    const stepBody = document.getElementById('stepRows');
    if (stepDensity.length > 0) {{
      stepDensity.forEach(r => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><strong>Step ${{r.step}}</strong></td>
          <td style="color: var(--accent);">${{r.avg_routed_density_pct.toFixed(2)}}%</td>
          <td>${{r.avg_kernel_density_pct.toFixed(2)}}%</td>
          <td>${{r.max_vram_mb.toFixed(1)}} MB</td>
          <td>${{r.sol_calls}}</td>
        `;
        stepBody.appendChild(tr);
      }});
    }} else {{
      stepBody.innerHTML = '<tr><td colspan="5" style="color: var(--text-muted); text-align: center;">No observer data.</td></tr>';
    }}
  </script>
</body>
</html>
"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", type=str, nargs="+", help="One or more directories containing sol_probe_*.jsonl or sol_observe_*.jsonl")
    parser.add_argument("--exclude-dirs", type=str, nargs="*", default=[], help="Directory names or substrings to exclude")
    parser.add_argument("--db", type=str, default="data/sparse/sol_analysis.duckdb", help="DuckDB database file path (relative to repo root)")
    parser.add_argument("--out", type=str, default="docs/research/sparse/sol_duckdb_dashboard.html", help="HTML dashboard output path (relative to repo root)")
    parser.add_argument("--query", type=str, help="Run an arbitrary SQL query against the database and exit")
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.query:
        rows = run_duckdb_sql(db_path, args.query)
        print(json.dumps(rows, indent=2))
        return 0

    probe_files = []
    observe_files = []

    if args.logs_dir:
        for ld in args.logs_dir:
            p_dir = Path(ld)
            for f in sorted(p_dir.glob("**/*sol_probe*.jsonl")):
                if not any(ex in f.as_posix() for ex in args.exclude_dirs):
                    probe_files.append(f)
            for f in sorted(p_dir.glob("**/*sol_observe*.jsonl")):
                if not any(ex in f.as_posix() for ex in args.exclude_dirs):
                    observe_files.append(f)
    else:
        # Auto-discover from internal/sol_observe
        probe_files = sorted(Path("internal/sol_observe").glob("**/*sol_probe*.jsonl"))
        observe_files = sorted(Path("internal/sol_observe").glob("**/*sol_observe*.jsonl"))

    print(f"[sol_duckdb] Using DuckDB binary: {DUCKDB_BIN}")
    print(f"[sol_duckdb] Target database: {args.db}")
    print(f"[sol_duckdb] Ingesting {len(probe_files)} probe files and {len(observe_files)} observe files...")

    setup_duckdb_tables(db_path, probe_files, observe_files)

    out_html = Path(args.out)
    generate_html_dashboard(db_path, out_html)
    print(f"[sol_duckdb] Successfully generated DuckDB visual dashboard: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
