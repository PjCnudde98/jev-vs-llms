"""Build the HTML report from one or more results folders.

    uv run jev-eval report results/<run-a> [results/<run-b> ...] -o report/index.html

The page is self-contained apart from Plotly (loaded from cdnjs, pinned). Each run
gets its own section; when several runs are passed (for example the stress runs), a
comparison section is added at the top. A human-written verdict is read from
`results/<run>/verdict.md` if it exists.

Chart rules follow the data-viz method used across the report: arms are coloured by
family (Jev, LLM, cascade, baseline) so the chart never needs more than three hues,
arms within a family differ by marker shape and a direct label, there is never a
second y-axis, and every chart has a table with the same numbers.
"""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from jev_eval import metrics as metrics_mod
from jev_eval.pricing import load_prices

PLOTLY = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.3/plotly.min.js"


def _family(name: str, kind: str | None, model: str | None = None) -> str:
    if name.startswith("cascade-"):
        return "cascade"
    if (model or "").startswith("jev"):
        return "jev"
    return {"jev": "jev", "anthropic": "llm", "openai": "llm", "baseline": "baseline"}.get(kind or "", "llm")


def _label(name: str) -> str:
    """'cascade-jev-to-opus-5.5@match_fallback' -> 'jev → opus-5.5 (match fallback)'."""
    if not name.startswith("cascade-"):
        return name
    pair, _, point = name.removeprefix("cascade-").partition("@")
    primary, _, fallback = pair.partition("-to-")
    return f"{primary} → {fallback}" + (f" ({point.replace('_', ' ')})" if point else "")


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def _money(x):
    if x is None:
        return "–"
    return f"${x:.2f}" if x >= 1 else f"${x:.4f}"


def _sec(x):
    return "–" if x is None else (f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s")


def _esc(x) -> str:
    return html.escape(str(x))


def _paragraphs(text: str) -> str:
    return "".join(f"<p>{_esc(p.strip())}</p>" for p in text.split("\n\n") if p.strip())


def _run_payload(run_dir: Path) -> dict:
    m = metrics_mod.compute(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    kinds = {name: a.get("kind") for name, a in manifest.get("arms", {}).items()}
    field = m["task"]["fields"][0]["name"]
    arms = []
    for name, a in m["splits"].get("eval", {}).items():
        f = a["fields"][field]
        arms.append(
            {
                "name": name,
                "label": _label(name),
                "family": _family(name, kinds.get(name), a["model"]),
                "model": a["model"],
                "versions": a["model_versions"],
                "accuracy": f["accuracy"],
                "ci": f["accuracy_ci"],
                "macro_f1": f["macro_f1"],
                "ece": f["calibration"]["ece"],
                "brier": f["calibration"]["brier"],
                "bins": f["calibration"]["bins"],
                "coverage": f["confidence_coverage"],
                "cost_per_1k": a["cost_per_1k"],
                "cost_per_1k_correct": a["cost_per_1k_correct"],
                "p50": a["latency"]["p50"],
                "p95": a["latency"]["p95"],
                "invalid": a["invalid_rate"],
                "consistency": f["consistency"],
                "oos": f.get("out_of_scope"),
                "per_class": {k: v["f1"] for k, v in f["per_class"].items()},
                "errors": a["error_examples"][field],
                "n": f["n"],
            }
        )
    arms.sort(key=lambda a: -(a["accuracy"] or 0))
    verdict = run_dir / "verdict.md"
    return {
        "id": run_dir.name,
        "dataset": m["dataset"],
        "field": field,
        "n_labels": len(m["task"]["fields"][0]["labels"]),
        "manifest": manifest,
        "arms": arms,
        "cascades": m.get("cascades", {}),
        "verdict": verdict.read_text() if verdict.exists() else None,
    }


def _tiles(run: dict) -> str:
    jev = next((a for a in run["arms"] if a["family"] == "jev"), None)
    llms = [a for a in run["arms"] if a["family"] == "llm"]
    if not jev or not llms:
        return ""
    best = max(llms, key=lambda a: a["accuracy"] or 0)
    gap = (jev["accuracy"] - best["accuracy"]) * 100
    tiles = [
        ("Jev accuracy", _pct(jev["accuracy"]), f"{gap:+.1f} pts vs {best['name']}"),
    ]
    if jev["cost_per_1k"] and best["cost_per_1k"]:
        tiles.append(
            (
                "Cost per 1k",
                _money(jev["cost_per_1k"]),
                f"{best['cost_per_1k'] / jev['cost_per_1k']:.0f}x cheaper than {best['name']}",
            )
        )
    if jev["p50"] and best["p50"]:
        tiles.append(
            ("Median latency", _sec(jev["p50"]), f"{best['p50'] / jev['p50']:.1f}x faster than {best['name']}")
        )
    if jev["ece"] is not None:
        tiles.append(("Calibration error (ECE)", f"{jev['ece']:.3f}", "lower is better; 0 = perfectly calibrated"))
    return (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="tile-label">{_esc(label)}</div><div class="tile-value">{_esc(value)}</div>'
            f'<div class="tile-note">{_esc(note)}</div></div>'
            for label, value, note in tiles
        )
        + "</div>"
    )


def _main_table(run: dict) -> str:
    rows = "".join(
        f"<tr><td><span class='key {a['family']}'></span>{_esc(a['label'])}</td><td class='num'>{_pct(a['accuracy'])}"
        f"<span class='ci'> {_pct(a['ci'][0])}–{_pct(a['ci'][1])}</span></td>"
        f"<td class='num'>{'–' if a['macro_f1'] is None else f'{a["macro_f1"]:.3f}'}</td>"
        f"<td class='num'>{'–' if a['ece'] is None else f'{a["ece"]:.3f}'}</td>"
        f"<td class='num'>{_money(a['cost_per_1k'])}</td><td class='num'>{_money(a['cost_per_1k_correct'])}</td>"
        f"<td class='num'>{_sec(a['p50'])}</td><td class='num'>{_sec(a['p95'])}</td>"
        f"<td class='num'>{_pct(a['invalid'])}</td><td class='num'>{_pct(a['consistency'])}</td>"
        f"<td class='num'>{_pct(a['oos']['recall']) if a['oos'] else '–'}</td></tr>"
        for a in run["arms"]
    )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Arm</th><th>Accuracy (95% CI)</th><th>Macro-F1</th><th>ECE</th>"
        "<th>$ / 1k</th><th>$ / 1k correct</th><th>p50</th><th>p95</th><th>Invalid</th><th>Consistency</th>"
        f"<th>Out-of-scope recall</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _cascade_section(run: dict, rid: str) -> str:
    if not run["cascades"]:
        return ""
    parts = ["<h3>Cascade: Jev first, LLM when unsure</h3>"]
    for name, c in run["cascades"].items():
        ref = c["reference"]
        rows = "".join(
            f"<tr><td>{_esc(point)}</td><td class='num'>{p['threshold']:.2f}</td>"
            f"<td class='num'>{_pct(p['eval']['routed_share'])}</td><td class='num'>{_pct(p['eval']['accuracy'])}</td>"
            f"<td class='num'>{_money(p['eval']['cost_per_1k'])}</td><td class='num'>{_sec(p['eval']['latency_p95'])}</td></tr>"
            for point, p in c["operating_points"].items()
        )
        rows += (
            f"<tr class='ref'><td>{_esc(c['primary'])} alone</td><td></td><td class='num'>0%</td>"
            f"<td class='num'>{_pct(ref['primary_only']['accuracy'])}</td><td class='num'>{_money(ref['primary_only']['cost_per_1k'])}</td>"
            f"<td class='num'>{_sec(ref['primary_only']['latency_p95'])}</td></tr>"
            f"<tr class='ref'><td>{_esc(c['fallback'])} alone</td><td></td><td class='num'>–</td>"
            f"<td class='num'>{_pct(ref['fallback_only']['accuracy'])}</td><td class='num'>{_money(ref['fallback_only']['cost_per_1k'] - ref['primary_only']['cost_per_1k'])}</td>"
            f"<td class='num'>–</td></tr>"
        )
        parts.append(
            f"<p class='lead'>{_esc(c['primary'])} answers when its confidence is at or above the threshold; below it, "
            f"{_esc(c['fallback'])} answers. Thresholds were picked on the calibration split and are reported on eval.</p>"
            f"<div class='grid2'><div class='chart' id='{rid}-cascade-acc-{_esc(name)}'></div>"
            f"<div class='chart' id='{rid}-cascade-cost-{_esc(name)}'></div></div>"
            "<div class='table-wrap'><table><thead><tr><th>Operating point</th><th>Threshold</th><th>Routed to LLM</th>"
            f"<th>Accuracy</th><th>$ / 1k</th><th>p95</th></tr></thead><tbody>{rows}</tbody></table></div>"
        )
    return "".join(parts)


def _errors_section(run: dict) -> str:
    blocks = []
    for a in run["arms"]:
        if not a["errors"] or a["family"] == "cascade":
            continue
        rows = "".join(
            f"<tr><td>{_esc(e['text'])}</td><td>{_esc(e['gold'])}</td><td>{_esc(e['pred'] or e.get('error') or 'invalid')}</td>"
            f"<td class='num'>{'–' if e['confidence'] is None else f'{e["confidence"]:.2f}'}</td></tr>"
            for e in a["errors"]
        )
        blocks.append(
            f"<details><summary>{_esc(a['name'])}: most confident mistakes</summary><div class='table-wrap'><table>"
            f"<thead><tr><th>Input</th><th>Gold</th><th>Predicted</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table></div></details>"
        )
    return "".join(blocks)


def _method_section(run: dict) -> str:
    man = run["manifest"]
    prices = load_prices()
    arm_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{_esc(a.get('model'))}</td><td>{_esc(', '.join(a.get('model_versions', [])) or '–')}</td>"
        f"<td><code>{_esc(json.dumps(a.get('params', {})))}</code></td>"
        f"<td>{_esc(prices.get(a.get('model'), {}).get('source', '–'))} {_esc(prices.get(a.get('model'), {}).get('as_of', ''))}</td>"
        f"<td class='num'>{_money(a.get('new_spend_usd'))}</td></tr>"
        for name, a in man.get("arms", {}).items()
    )
    pkgs = ", ".join(f"{k} {v}" for k, v in man.get("packages", {}).items())
    return (
        "<details><summary>Method, versions and spend</summary>"
        f"<p>Run <code>{_esc(run['id'])}</code> on <strong>{_esc(run['dataset'])}</strong>, field "
        f"<code>{_esc(run['field'])}</code> with {run['n_labels']} labels. Started {_esc(man.get('started', '–'))}. "
        f"Packages: {_esc(pkgs)}.</p><div class='table-wrap'><table><thead><tr><th>Arm</th><th>Model</th><th>Version answered</th>"
        f"<th>Params</th><th>Price source</th><th>New spend</th></tr></thead><tbody>{arm_rows}</tbody></table></div></details>"
    )


def _run_section(run: dict, index: int) -> str:
    rid = f"r{index}"
    verdict = (
        _paragraphs(run["verdict"])
        if run["verdict"]
        else f"<p class='muted'>No verdict written yet. Add it to <code>results/{_esc(run['id'])}/verdict.md</code>.</p>"
    )
    n = run["arms"][0]["n"] if run["arms"] else 0
    return f"""
<section class="run" id="{rid}">
  <h2>{_esc(run["dataset"])}<span class="sub"> · {n} eval items · {run["n_labels"]} labels · run {_esc(run["id"])}</span></h2>
  <div class="verdict">{verdict}</div>
  {_tiles(run)}
  <h3>Head-to-head</h3>
  {_main_table(run)}
  <h3>Cost against accuracy</h3>
  <p class="lead">Up and to the left is better. Whiskers are 95% bootstrap intervals. Arms that run locally for free are left out of this chart.</p>
  <div class="chart tall" id="{rid}-scatter"></div>
  <h3>Latency per prediction</h3>
  <p class="lead">Dot is the median (p50), the line runs to p95. Log scale.</p>
  <div class="chart" id="{rid}-latency"></div>
  <h3>Can we trust the probabilities?</h3>
  <p class="lead">Each dot is a confidence bin: how sure the arm said it was (x) against how often it was right (y). On the diagonal is perfectly calibrated; below it is overconfident. Dot size is the number of predictions in the bin.</p>
  <div class="grid-small" id="{rid}-reliability"></div>
  {_cascade_section(run, rid)}
  <h3>Where each arm struggles</h3>
  <p class="lead">F1 per class for the 25 classes where the arms do worst on average.</p>
  <div class="chart tall" id="{rid}-heatmap"></div>
  {_errors_section(run)}
  {_method_section(run)}
</section>"""


def _comparison_section(runs: list[dict]) -> str:
    if len(runs) < 2:
        return ""
    return """
<section class="run" id="compare">
  <h2>Across runs</h2>
  <p class="lead">Accuracy of each arm in every run passed to this report, for example the clean set against its stress variants.</p>
  <div class="chart tall" id="compare-chart"></div>
</section>"""


CSS = """
:root{color-scheme:light;--surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--series-1:#2a78d6;--series-2:#eb6834;
--series-3:#1baf7a;--seq-100:#cde2fb;--seq-400:#3987e5;--seq-700:#0d366b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;
--text-primary:#f0efec;--text-secondary:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--seq-100:#0d366b;--seq-400:#3987e5;--seq-700:#cde2fb}}
:root[data-theme="dark"]{color-scheme:dark;--surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#f0efec;--text-secondary:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--series-1:#3987e5;--series-2:#d95926;
--series-3:#199e70;--seq-100:#0d366b;--seq-400:#3987e5;--seq-700:#cde2fb}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text-primary);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 16px 80px}
header h1{font-size:30px;line-height:1.2;margin:0 0 6px;font-weight:650}
header p{margin:0;color:var(--text-secondary)}
.banner{margin-top:16px;padding:10px 14px;border:1px solid var(--border);border-radius:8px;background:var(--surface-1);color:var(--text-secondary);font-size:13px}
section.run{margin-top:48px;padding-top:8px;border-top:1px solid var(--grid)}
h2{font-size:22px;margin:16px 0 8px;font-weight:650}
h2 .sub{font-size:14px;font-weight:400;color:var(--text-secondary)}
h3{font-size:16px;margin:36px 0 4px;font-weight:650}
.lead{margin:0 0 12px;color:var(--text-secondary);font-size:14px;max-width:72ch}
.muted{color:var(--muted)}
.verdict{font-size:16px;max-width:72ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile-label{font-size:13px;color:var(--text-secondary)}
.tile-value{font-size:28px;font-weight:650;margin:2px 0}
.tile-note{font-size:12.5px;color:var(--muted)}
.table-wrap{overflow-x:auto;background:var(--surface-1);border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-weight:600;color:var(--text-secondary);white-space:nowrap}
tr:last-child td{border-bottom:0}
td:first-child{white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.ci{color:var(--muted);font-size:12px}
tr.ref td{color:var(--text-secondary)}
.key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:baseline}
.key.jev{background:var(--series-1)}.key.llm{background:var(--series-2)}.key.cascade{background:var(--series-3)}.key.baseline{background:var(--muted)}
.chart{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;height:340px}
.chart.tall{height:460px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:12px}
.grid-small{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.grid-small .chart{height:240px}
.grid2>*,.grid-small>*{min-width:0;overflow:hidden}
details{margin-top:16px}
summary{cursor:pointer;font-weight:600;padding:6px 0}
details p{color:var(--text-secondary);font-size:14px}
code{font-size:12.5px;background:var(--grid);padding:1px 5px;border-radius:4px}
.legend-row{display:flex;gap:18px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary);margin:8px 0 0}
"""

JS = r"""
const R = JSON.parse(document.getElementById('report-data').textContent);
const SYMBOLS = ['circle','diamond','square','triangle-up','cross','x'];
function tok(){const s=getComputedStyle(document.documentElement);return k=>s.getPropertyValue(k).trim();}
function famColor(t,f){return {jev:t('--series-1'),llm:t('--series-2'),cascade:t('--series-3'),baseline:t('--muted')}[f]||t('--muted');}
function symbolsFor(arms){const seen={},out={};for(const a of arms){seen[a.family]=(seen[a.family]||0);out[a.name]=SYMBOLS[seen[a.family]%SYMBOLS.length];seen[a.family]++;}return out;}
function layout(t,extra){return Object.assign({paper_bgcolor:t('--surface-1'),plot_bgcolor:t('--surface-1'),
 font:{family:'system-ui,-apple-system,Segoe UI,sans-serif',size:12,color:t('--text-secondary')},
 margin:{l:64,r:24,t:24,b:52},showlegend:false,hoverlabel:{bgcolor:t('--surface-1'),bordercolor:t('--axis'),font:{color:t('--text-primary')}}},extra);}
function ax(t,extra){return Object.assign({gridcolor:t('--grid'),linecolor:t('--axis'),zeroline:false,tickfont:{color:t('--muted')},title:{font:{color:t('--text-secondary')}}},extra);}
const CFG={displayModeBar:false,responsive:true};
function pct(x){return (100*x).toFixed(1)+'%';}

function drawRun(run,i,t){
  const rid='r'+i, sym=symbolsFor(run.arms);
  // Cost vs accuracy
  const paid=run.arms.filter(a=>a.cost_per_1k>0);
  Plotly.react(rid+'-scatter',paid.map(a=>({type:'scatter',mode:'markers+text',x:[a.cost_per_1k],y:[100*a.accuracy],
    text:[a.label],textposition:'top center',textfont:{color:t('--text-primary'),size:12},
    error_y:{type:'data',symmetric:false,array:[100*(a.ci[1]-a.accuracy)],arrayminus:[100*(a.accuracy-a.ci[0])],color:famColor(t,a.family),thickness:1.5,width:0},
    marker:{size:12,color:famColor(t,a.family),symbol:sym[a.name],line:{color:t('--surface-1'),width:2}},
    hovertemplate:`<b>${a.label}</b><br>Accuracy %{y:.1f}%<br>$%{x:.4f} per 1k<extra></extra>`})),
    layout(t,{xaxis:ax(t,{type:'log',dtick:1,tickprefix:'$',title:{text:'$ per 1,000 predictions (log scale)'}}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
  // Latency
  const lat=run.arms.filter(a=>a.p50);
  Plotly.react(rid+'-latency',lat.map(a=>({type:'scatter',mode:'lines+markers',x:[a.p50,a.p95],y:[a.label,a.label],
    line:{color:famColor(t,a.family),width:2},marker:{size:[11,7],color:famColor(t,a.family),symbol:sym[a.name],line:{color:t('--surface-1'),width:2}},
    hovertemplate:`<b>${a.label}</b><br>p50 ${(a.p50).toFixed(3)} s<br>p95 ${(a.p95).toFixed(3)} s<extra></extra>`})),
    layout(t,{margin:{l:230,r:24,t:16,b:52},xaxis:ax(t,{type:'log',dtick:1,ticksuffix:' s',title:{text:'Seconds per prediction (log scale)'}}),yaxis:ax(t,{autorange:'reversed',gridcolor:t('--grid')})}),CFG);
  // Reliability small multiples
  const box=document.getElementById(rid+'-reliability'); box.innerHTML='';
  // Create every panel before plotting: an auto-fit grid collapses empty tracks, so a
  // panel measured while alone would take the whole row.
  const rel=run.arms.filter(a=>a.bins&&a.bins.length&&a.family!=='cascade');
  const divs=rel.map((a,j)=>{const d=document.createElement('div');d.className='chart';d.id=`${rid}-rel-${j}`;box.appendChild(d);return d;});
  rel.forEach((a,j)=>{
    const d=divs[j];
    const n=Math.max(...a.bins.map(b=>b.n));
    Plotly.react(d.id,[{type:'scatter',mode:'lines',x:[0,1],y:[0,1],line:{color:t('--axis'),width:1},hoverinfo:'skip'},
      {type:'scatter',mode:'lines+markers',x:a.bins.map(b=>b.confidence),y:a.bins.map(b=>b.accuracy),
       line:{color:famColor(t,a.family),width:2},marker:{size:a.bins.map(b=>6+12*Math.sqrt(b.n/n)),color:famColor(t,a.family),line:{color:t('--surface-1'),width:2}},
       customdata:a.bins.map(b=>b.n),hovertemplate:'Said %{x:.2f}, right %{y:.2f}<br>%{customdata} predictions<extra></extra>'}],
      layout(t,{width:d.clientWidth,height:d.clientHeight,margin:{l:44,r:12,t:34,b:38},title:{text:`${a.name} · ECE ${a.ece==null?'–':a.ece.toFixed(3)}`,font:{size:13,color:t('--text-primary')},x:0.04},
        xaxis:ax(t,{range:[0,1.02],dtick:0.25}),yaxis:ax(t,{range:[0,1.02],dtick:0.25})}),CFG);
  });
  // Cascade
  for(const [name,c] of Object.entries(run.cascades||{})){
    const s=c.sweep.eval, x=s.map(p=>100*p.routed_share);
    const pts=Object.entries(c.operating_points).map(([k,p])=>({k,x:100*p.eval.routed_share,acc:100*p.eval.accuracy,cost:p.eval.cost_per_1k}));
    const mk=(y,yp,title,suffix,fmt)=>[
      {type:'scatter',mode:'lines',x,y,line:{color:t('--series-3'),width:2},customdata:s.map(p=>p.threshold),
       hovertemplate:`Threshold %{customdata:.2f}<br>Routed %{x:.1f}%<br>${title} %{y:${fmt}}${suffix}<extra></extra>`},
      {type:'scatter',mode:'markers+text',x:pts.map(p=>p.x),y:yp,text:pts.map(p=>p.k.replace(/_/g,' ')),textposition:pts.map((p,j)=>j%2?'bottom right':'top right'),
       textfont:{color:t('--text-primary'),size:12},marker:{size:10,color:t('--series-3'),line:{color:t('--surface-1'),width:2}},hoverinfo:'skip'}];
    Plotly.react(`${rid}-cascade-acc-${name}`,mk(s.map(p=>100*p.accuracy),pts.map(p=>p.acc),'Accuracy','%','.1f'),
      layout(t,{xaxis:ax(t,{title:{text:'Share routed to the LLM (%)'},ticksuffix:'%'}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
    Plotly.react(`${rid}-cascade-cost-${name}`,mk(s.map(p=>p.cost_per_1k),pts.map(p=>p.cost),'Cost $','','.4f'),
      layout(t,{xaxis:ax(t,{title:{text:'Share routed to the LLM (%)'},ticksuffix:'%'}),yaxis:ax(t,{title:{text:'$ per 1,000 predictions'},tickprefix:'$'})}),CFG);
  }
  // Heatmap of the hardest classes
  const arms=run.arms.filter(a=>!a.name.startsWith('cascade-'));
  const classes=Object.keys(arms[0]?.per_class||{});
  const mean=c=>arms.reduce((s,a)=>s+(a.per_class[c]??0),0)/arms.length;
  const worst=classes.sort((a,b)=>mean(a)-mean(b)).slice(0,25);
  Plotly.react(rid+'-heatmap',[{type:'heatmap',x:arms.map(a=>a.label),y:worst,z:worst.map(c=>arms.map(a=>a.per_class[c]??null)),
    zmin:0,zmax:1,colorscale:[[0,t('--seq-100')],[0.5,t('--seq-400')],[1,t('--seq-700')]],xgap:2,ygap:2,
    colorbar:{title:{text:'F1',font:{color:t('--text-secondary')}},tickfont:{color:t('--muted')},outlinewidth:0,thickness:10},
    hovertemplate:'%{y}<br>%{x}: F1 %{z:.2f}<extra></extra>'}],
    layout(t,{margin:{l:220,r:24,t:16,b:90},xaxis:ax(t,{gridcolor:'rgba(0,0,0,0)'}),yaxis:ax(t,{autorange:'reversed',gridcolor:'rgba(0,0,0,0)',tickfont:{size:11,color:t('--text-secondary')}})}),CFG);
}

function drawCompare(t){
  if(R.runs.length<2) return;
  const names=[...new Set(R.runs.flatMap(r=>r.arms.map(a=>a.name)))];
  const fam={}; R.runs.forEach(r=>r.arms.forEach(a=>fam[a.name]=a.family));
  const sym=symbolsFor(names.map(n=>({name:n,family:fam[n]})));
  Plotly.react('compare-chart',names.map(n=>({type:'scatter',mode:'lines+markers',name:n,
    x:R.runs.map(r=>r.id),y:R.runs.map(r=>{const a=r.arms.find(a=>a.name===n);return a?100*a.accuracy:null;}),
    line:{color:famColor(t,fam[n]),width:2},marker:{size:9,color:famColor(t,fam[n]),symbol:sym[n],line:{color:t('--surface-1'),width:2}},
    hovertemplate:`<b>${n}</b><br>%{x}<br>Accuracy %{y:.1f}%<extra></extra>`})),
    layout(t,{showlegend:true,legend:{orientation:'h',y:-0.25,font:{color:t('--text-secondary')}},margin:{l:64,r:24,t:16,b:120},
      xaxis:ax(t,{}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
}

function drawAll(){const t=tok();R.runs.forEach((r,i)=>drawRun(r,i,t));drawCompare(t);}
drawAll();
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',drawAll);
new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
"""


def build(run_dirs: list[Path], out: Path, title: str = "Jev vs LLMs") -> Path:
    runs = [_run_payload(d) for d in run_dirs]
    versions = sorted({v for r in runs for a in r["arms"] if a["family"] == "jev" for v in a["versions"]})
    spend = sum(r["manifest"].get("new_spend_usd", 0) for r in runs)
    banner = (
        f"Jev version: {', '.join(versions) or '–'} · Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · "
        f"API spend for these runs: {_money(spend)}"
    )
    data = json.dumps({"runs": runs}).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{CSS}</style>
<script src="{PLOTLY}"></script></head>
<body><main>
<header><h1>{_esc(title)}</h1>
<p>Quality, cost, latency and calibration of Typesafe's Jev against a cheap and a frontier LLM, on the same data and inputs.</p>
<div class="banner">{_esc(banner)}</div>
<div class="legend-row"><span><span class="key jev"></span>Jev</span><span><span class="key llm"></span>LLM</span>
<span><span class="key cascade"></span>Cascade</span><span><span class="key baseline"></span>Classic baseline</span></div></header>
{_comparison_section(runs)}
{"".join(_run_section(r, i) for i, r in enumerate(runs))}
</main>
<script type="application/json" id="report-data">{data}</script>
<script>{JS}</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
