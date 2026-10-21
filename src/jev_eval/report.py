"""Build the HTML report from one or more results folders.

    uv run jev-eval report results/<run-a> [results/<run-b> ...] -o report/index.html

The page reads as a story: a headline worked out from the runs, the evidence chapter by chapter
(accuracy, cost, cascade, languages), a calculator for the price of a mistake and a verdict. A
deep dive below holds every table and chart, one tab per run. It is self-contained apart from
Plotly (cdnjs, pinned) and the Inter and Roboto Mono fonts. A human-written verdict is read from
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
from itertools import combinations
from pathlib import Path

from jev_eval import business
from jev_eval import metrics as metrics_mod
from jev_eval.pricing import load_prices
from jev_eval.types import read_jsonl

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


def _eval_rows(run_dir: Path, arm: str) -> list[dict]:
    path = run_dir / "eval" / f"{arm}.jsonl"
    return read_jsonl(path) if path.exists() else []


def _language_of(dataset: str) -> str | None:
    """The language recorded in the dataset's items (`meta.language`), if any."""
    path = metrics_mod.ROOT / "data" / "processed" / dataset / "eval.jsonl"
    if not path.exists():
        return None
    with open(path) as f:
        first = f.readline()
    return json.loads(first).get("meta", {}).get("language") if first.strip() else None


def _pairs(run_dir: Path, arms: list[dict], field: str) -> list[dict]:
    """Paired tests between Jev and every other arm (or all pairs when there is no Jev arm)."""
    base = [a for a in arms if a["family"] != "cascade"]
    jev = [a for a in base if a["family"] == "jev"]
    pairs = [(j, b) for j in jev for b in base if b is not j] if jev else list(combinations(base, 2))
    out = []
    for a, b in pairs:
        rows_a, rows_b = _eval_rows(run_dir, a["name"]), _eval_rows(run_dir, b["name"])
        if rows_a and rows_b:
            out.append({"a": a["name"], "b": b["name"], **metrics_mod.paired_difference(rows_a, rows_b, field)})
    return out


def _headline(run_dir: Path, arms: list[dict], cascades: dict, field: str) -> dict | None:
    """Jev against the best LLM: the one line a reader wants from a run."""
    jev = max((a for a in arms if a["family"] == "jev"), key=lambda a: a["accuracy"] or 0, default=None)
    llm = max((a for a in arms if a["family"] == "llm"), key=lambda a: a["accuracy"] or 0, default=None)
    if not jev or not llm:
        return None
    rows_j, rows_l = _eval_rows(run_dir, jev["name"]), _eval_rows(run_dir, llm["name"])
    out = {
        "jev": jev["name"],
        "llm": llm["name"],
        "jev_accuracy": jev["accuracy"],
        "llm_accuracy": llm["accuracy"],
        "jev_cost_per_1k": jev["cost_per_1k"],
        "llm_cost_per_1k": llm["cost_per_1k"],
        "paired": metrics_mod.paired_difference(rows_j, rows_l, field) if rows_j and rows_l else None,
        "cascade": None,
    }
    for c in cascades.values():
        point = c["operating_points"].get("match_fallback")
        if point and c["fallback"] == llm["name"]:
            e = point["eval"]
            out["cascade"] = {
                "routed_share": e["routed_share"],
                "accuracy": e["accuracy"],
                "cost_per_1k": e["cost_per_1k"],
            }
            break
    return out


def _languages(runs: list[dict]) -> dict | None:
    """Accuracy per arm and language, with the paired change against English (or the first language)."""
    by_language: dict[str, dict] = {}
    for r in runs:
        if r["language"] and r["language"] not in by_language:
            by_language[r["language"]] = r
    if len(by_language) < 2:
        return None
    order = sorted(by_language, key=lambda lang: (lang != "en", list(by_language).index(lang)))
    reference = order[0]
    names = [
        n
        for n in dict.fromkeys(a["name"] for r in by_language.values() for a in r["arms"] if a["family"] != "cascade")
        if sum(any(a["name"] == n for a in by_language[lang]["arms"]) for lang in order) >= 2
    ]
    arms: dict[str, dict] = {}
    for name in names:
        cells = {}
        for lang in order:
            run = by_language[lang]
            arm = next((a for a in run["arms"] if a["name"] == name), None)
            if not arm:
                continue
            cell = {
                "accuracy": arm["accuracy"],
                "ci": arm["ci"],
                "cost_per_1k_correct": arm["cost_per_1k_correct"],
                "ece": arm["ece"],
                "family": arm["family"],
                "delta": None,
            }
            ref_run = by_language[reference]
            if lang != reference and any(a["name"] == name for a in ref_run["arms"]):
                rows, ref_rows = _eval_rows(Path(run["dir"]), name), _eval_rows(Path(ref_run["dir"]), name)
                if rows and ref_rows:
                    cell["delta"] = metrics_mod.paired_difference(rows, ref_rows, run["field"])
            cells[lang] = cell
        arms[name] = cells
    return {"languages": order, "reference": reference, "arms": arms}


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
    cascades = m.get("cascades", {})
    return {
        "id": run_dir.name,
        "dir": str(run_dir),
        "dataset": m["dataset"],
        "language": _language_of(m["dataset"]),
        "field": field,
        "n_labels": len(m["task"]["fields"][0]["labels"]),
        "manifest": manifest,
        "arms": arms,
        "cascades": cascades,
        "verdict": verdict.read_text() if verdict.exists() else None,
        "business": business.analyse(business.arms_from_metrics(m)),
        "pairs": _pairs(run_dir, arms, field),
        "headline": _headline(run_dir, arms, cascades, field),
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


def _pts(x: float) -> str:
    return f"{100 * x:+.1f} pts"


def _p(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _paired_text(d: dict) -> str:
    lo, hi = d["difference_ci"]
    return f"{_pts(d['difference'])} (95% CI {100 * lo:+.1f} to {100 * hi:+.1f}), {_p(d['p_value'])}"


def _verdict_word(d: dict) -> str:
    if d["p_value"] >= 0.05:
        return "no significant difference"
    return "significantly higher" if d["difference"] > 0 else "significantly lower"


def _summary_section(runs: list[dict]) -> str:
    rows, notes = [], []
    for r in runs:
        h = r["headline"]
        if not h:
            continue
        label = r["dataset"] + (f" ({r['language']})" if r["language"] else "")
        ratio = f"{h['llm_cost_per_1k'] / h['jev_cost_per_1k']:.0f}x" if h["jev_cost_per_1k"] else "–"
        gap = "–" if not h["paired"] else _paired_text(h["paired"])
        cascade = "–"
        if h["cascade"]:
            c = h["cascade"]
            saving = 100 * (1 - c["cost_per_1k"] / h["llm_cost_per_1k"]) if h["llm_cost_per_1k"] else 0
            cascade = f"{_pct(c['accuracy'])}, {_pct(c['routed_share'])} to LLM, {saving:.0f}% cheaper than {h['llm']}"
        rows.append(
            f"<tr><td>{_esc(label)}</td><td class='num'>{_pct(h['jev_accuracy'])}</td>"
            f"<td class='num'>{_pct(h['llm_accuracy'])} <span class='ci'>{_esc(h['llm'])}</span></td><td>{_esc(gap)}</td>"
            f"<td class='num'>{_money(h['jev_cost_per_1k'])}</td><td class='num'>{_money(h['llm_cost_per_1k'])}</td>"
            f"<td class='num'>{ratio}</td><td>{_esc(cascade)}</td></tr>"
        )
        if h["paired"]:
            notes.append(
                f"<li><strong>{_esc(label)}:</strong> {_esc(h['jev'])} is {_verdict_word(h['paired'])} than {_esc(h['llm'])} "
                f"({_esc(_paired_text(h['paired']))}) at {ratio} lower cost per 1k calls.</li>"
            )
    if not rows:
        return ""
    return (
        "<div class='deep-block' id='all-numbers'><h3>Summary table</h3>"
        "<p class='lead'>Jev against the most accurate LLM in each run. The gap is Jev minus the LLM on the same items; "
        "p is an exact McNemar test, and a p of 0.05 or more means the data cannot tell the two apart. "
        "Public and in-house sets should be read separately.</p>"
        f"<ul class='takeaways'>{''.join(notes)}</ul>"
        "<div class='table-wrap'><table><thead><tr><th>Run</th><th>Jev accuracy</th><th>Best LLM</th>"
        "<th>Gap (Jev minus LLM)</th><th>Jev $ / 1k</th><th>LLM $ / 1k</th><th>Cost ratio</th>"
        f"<th>Cascade (match fallback)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def _significance_section(run: dict) -> str:
    if not run["pairs"]:
        return ""
    rows = "".join(
        f"<tr><td>{_esc(p['a'])} vs {_esc(p['b'])}</td><td class='num'>{_pct(p['accuracy_a'])} vs {_pct(p['accuracy_b'])}</td>"
        f"<td>{_esc(_paired_text(p))}</td><td class='num'>{p['only_a_right']} / {p['only_b_right']}</td>"
        f"<td>{_verdict_word(p)}</td></tr>"
        for p in run["pairs"]
    )
    return (
        "<h3>Is the difference real?</h3>"
        "<p class='lead'>Paired comparison on the same items. A p of 0.05 or more means the gap could be chance at this "
        "sample size. The two counts are the items only the first arm got right and only the second arm got right.</p>"
        "<div class='table-wrap'><table><thead><tr><th>Pair</th><th>Accuracy</th><th>Difference</th>"
        f"<th>Only A / only B right</th><th>Verdict for A</th></tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _breakeven_text(v: dict | None) -> str:
    if v is None:
        return "reference"
    be = v["break_even_cost_per_error"]
    return {
        "dominates": "better on cost and accuracy",
        "dominated": "worse on cost and accuracy",
        "cheaper_but_less_accurate": f"cheaper overall if a mistake costs under ${be:.2f}" if be is not None else "–",
        "costlier_but_more_accurate": f"cheaper overall if a mistake costs over ${be:.2f}" if be is not None else "–",
    }[v["verdict"]]


def _business_section(run: dict, rid: str) -> str:
    b = run["business"]
    if not b["arms"]:
        return ""
    volumes = b["volumes"]
    head = "".join(f"<th>API $ / month at {v:,}</th>" for v in volumes)
    rows = ""
    for a in b["arms"]:
        v = a["vs_reference"]
        vs = (
            "reference"
            if v is None
            else f"{v['accuracy_points']:+.1f} pts, "
            + ("n/a" if v["saving_pct"] is None else f"{v['saving_pct']:.0f}% cheaper")
        )
        cells = "".join(f"<td class='num'>{_money(a['monthly'][str(vol)]['api_usd'])}</td>" for vol in volumes)
        rows += (
            f"<tr><td>{_esc(_label(a['name']))}</td><td class='num'>{_pct(a['accuracy'])}</td>"
            f"<td class='num'>{a['errors_per_1k']:.0f}</td>{cells}<td>{_esc(vs)}</td>"
            f"<td>{_esc(_breakeven_text(v))}</td><td>{_esc(', '.join(a['dominated_by']) or '–')}</td></tr>"
        )
    ref = b["reference"]
    return f"""
  <h3>What it costs at scale</h3>
  <p class="lead">API calls only; the local baseline shows $0 because its hosting is not measured. Break-even is the price of one
  wrong answer at which an arm costs the same overall as {_esc(ref or "the reference")}. An arm marked as dominated is beaten on
  both cost and accuracy by another arm, so there is no reason to pick it.</p>
  <div class="table-wrap"><table><thead><tr><th>Arm</th><th>Accuracy</th><th>Mistakes / 1k</th>{head}
  <th>vs {_esc(ref or "reference")}</th><th>Break-even</th><th>Dominated by</th></tr></thead><tbody>{rows}</tbody></table></div>
  <h3>Your numbers</h3>
  <p class="lead">Enter your volume and what one wrong answer costs you (a manual fix, a refund, a lost customer).
  The cheapest all-in option is highlighted.</p>
  <div class="calc"><label>Calls per month <input id="{rid}-vol" type="number" min="0" step="1000" value="100000"></label>
  <label>Cost of one mistake (USD) <input id="{rid}-cpe" type="number" min="0" step="0.1" value="1"></label></div>
  <div class="table-wrap"><table><thead><tr><th>Arm</th><th>API $ / month</th><th>Mistakes / month</th>
  <th>Cost of mistakes</th><th>All-in $ / month</th></tr></thead><tbody id="{rid}-calc"></tbody></table></div>"""


def _languages_section(langs: dict | None) -> str:
    if not langs:
        return ""
    order, ref = langs["languages"], langs["reference"]
    head = "".join(f"<th>{_esc(lang)}</th>" for lang in order)
    acc_rows, cost_rows = "", ""
    for name, cells in langs["arms"].items():
        row = ""
        for lang in order:
            c = cells.get(lang)
            if not c:
                row += "<td class='num'>–</td>"
                continue
            extra = ""
            if c["delta"]:
                d = c["delta"]
                extra = f"<br><span class='ci'>{_pts(d['difference'])} vs {_esc(ref)}, {_p(d['p_value'])}</span>"
            row += f"<td class='num'>{_pct(c['accuracy'])}{extra}</td>"
        acc_rows += f"<tr><td>{_esc(name)}</td>{row}</tr>"
        cost_rows += (
            f"<tr><td>{_esc(name)}</td>"
            + "".join(
                f"<td class='num'>{_money(cells[lang]['cost_per_1k_correct']) if lang in cells else '–'}</td>"
                for lang in order
            )
            + "</tr>"
        )
    return f"""
<div class="deep-block" id="by-language">
  <h3>By language</h3>
  <p class="lead">The same items in every language, so a change is down to the language. Each cell under the first
  language shows the paired change against {_esc(ref)} for the same arm; p below 0.05 means the drop is unlikely to be chance.</p>
  <div class="table-wrap"><table><thead><tr><th>Arm</th>{head}</tr></thead><tbody>{acc_rows}</tbody></table></div>
  <h3 class='h4'>Cost per 1,000 correct answers</h3>
  <p class="lead">Cost divided by correct answers: a language where an arm is less accurate costs more per useful answer.</p>
  <div class="table-wrap"><table><thead><tr><th>Arm</th>{head}</tr></thead><tbody>{cost_rows}</tbody></table></div>
</div>"""


def _run_section(run: dict, index: int) -> str:
    rid = f"r{index}"
    verdict = (
        _paragraphs(run["verdict"])
        if run["verdict"]
        else f"<p class='muted'>No verdict written yet. Add it to <code>results/{_esc(run['id'])}/verdict.md</code>.</p>"
    )
    n = run["arms"][0]["n"] if run["arms"] else 0
    return f"""
<section class="run-panel" id="{rid}" data-run="{index}" role="tabpanel">
  <h3 class="run-title">{_esc(run["dataset"])}<span class="sub"> · {n} eval items · {run["n_labels"]} labels · run {_esc(run["id"])}</span></h3>
  <div class="verdict">{verdict}</div>
  {_tiles(run)}
  <h3>Head-to-head</h3>
  {_main_table(run)}
  {_significance_section(run)}
  {_business_section(run, rid)}
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
<div class="deep-block" id="compare">
  <h3>Across runs</h3>
  <p class="lead">Accuracy of each arm in every run passed to this report, for example the clean set against its stress variants.</p>
  <div class="chart tall" id="compare-chart"></div>
</div>"""


# --- The story: the few numbers a reader needs, worked out from the runs ---------------------------

LANGUAGE_NAMES = {"en": "English", "fr": "French", "nl": "Dutch", "ja": "Japanese", "de": "German", "es": "Spanish"}


def _story_label(name: str) -> str:
    """'cascade-jev-to-gpt-5.6-sol@match_fallback' -> 'jev first, then gpt-5.6-sol'."""
    if not name.startswith("cascade-"):
        return name
    primary, _, fallback = name.removeprefix("cascade-").partition("@")[0].partition("-to-")
    return f"{primary} first, then {fallback}"


def _price(x: float) -> str:
    """A break-even price per mistake, with as many digits as it needs."""
    if x >= 10:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:.2f}"
    return f"${x:.2g}"


def _span(values: list[float], fmt: str = "{:.1f}") -> str:
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def _about(values: list[float]) -> str:
    """Whole numbers for a headline: '5' or '5–7'."""
    lo, hi = round(min(values)), round(max(values))
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _distinct(arms: list[dict]) -> list[dict]:
    """Business arms without duplicates (a cascade that never routes is just its primary)."""
    seen, out = set(), []
    for a in sorted(arms, key=lambda a: a["name"].startswith("cascade-")):
        key = (round(a["cost_per_1k"], 6), round(a["accuracy"], 6))
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


def _envelope(arms: list[dict]) -> list[dict]:
    """Which arm is cheapest all-in (API cost plus mistakes) as the price of one mistake grows.

    Returns consecutive ranges [{name, from, to}] in USD per mistake; the last range has no upper end.
    """
    arms = _distinct(arms)
    if not arms:
        return []
    cur = min(arms, key=lambda a: (a["cost_per_1k"], -a["accuracy"]))
    start, out = 0.0, []
    while True:
        nxt = None
        for b in arms:
            if b["accuracy"] > cur["accuracy"]:
                c = (b["cost_per_1k"] - cur["cost_per_1k"]) / 1000 / (b["accuracy"] - cur["accuracy"])
                if c >= start - 1e-12 and (
                    nxt is None
                    or c < nxt[0] - 1e-12
                    or (abs(c - nxt[0]) <= 1e-12 and b["accuracy"] > nxt[1]["accuracy"])
                ):
                    nxt = (c, b)
        if not nxt or nxt[0] > start + 1e-12:
            out.append({"name": cur["name"], "from": start, "to": nxt[0] if nxt else None})
        if not nxt:
            return out
        start, cur = max(nxt[0], start), nxt[1]


def _story(runs: list[dict], langs: dict | None) -> dict | None:
    """Headline, findings and decision ranges across the runs that have both Jev and an LLM."""
    picks = [(i, r) for i, r in enumerate(runs) if r["headline"]]
    if not picks:
        return None
    hs = [r["headline"] for _, r in picks]
    jev, llm = hs[0]["jev"], hs[0]["llm"]
    by_lang = all(r["language"] for _, r in picks) and len({r["language"] for _, r in picks}) == len(picks)
    unit = "languages" if by_lang else "runs"

    def arm(run, name):
        return next((a for a in run["arms"] if a["name"] == name), None)

    tabs = [
        {
            "index": i,
            "code": (r["language"] or r["id"]).upper() if by_lang else r["id"],
            "name": LANGUAGE_NAMES.get(r["language"], r["language"]) if by_lang else r["dataset"],
        }
        for i, r in picks
    ]
    gaps = [100 * (h["jev_accuracy"] - h["llm_accuracy"]) for h in hs]
    sig = sum(1 for h in hs if h["paired"] and h["paired"]["p_value"] < 0.05)
    ratios = [h["llm_cost_per_1k"] / h["jev_cost_per_1k"] for h in hs if h["jev_cost_per_1k"]]
    speeds = [arm(r, llm)["p50"] / arm(r, jev)["p50"] for _, r in picks if arm(r, jev)["p50"] and arm(r, llm)["p50"]]

    # The cheaper LLM, if the runs have one, and Jev's paired gap to it.
    cheap = []
    for _, r in picks:
        others = [a for a in r["arms"] if a["family"] == "llm" and a["name"] != llm and a["cost_per_1k"]]
        if others:
            c = min(others, key=lambda a: a["cost_per_1k"])
            pair = next((p for p in r["pairs"] if p["a"] == jev and p["b"] == c["name"]), None)
            if pair:
                cheap.append((c["name"], pair))
    cheap_name = cheap[0][0] if cheap else None

    cascades = [(h, h["cascade"]) for h in hs if h["cascade"]]
    cascade_name = next(
        (
            a["name"]
            for _, r in picks
            for a in r["arms"]
            if a["family"] == "cascade" and a["name"].endswith("@match_fallback")
        ),
        None,
    )

    lang_deltas = []
    if langs and jev in langs["arms"]:
        for lang in langs["languages"]:
            d = langs["arms"][jev].get(lang, {}).get("delta")
            if d:
                lang_deltas.append({"lang": lang, "name": LANGUAGE_NAMES.get(lang, lang), **d})

    baseline_gaps = [
        100 * (b["accuracy"] - arm(r, jev)["accuracy"])
        for _, r in picks
        if (b := next((a for a in r["arms"] if a["family"] == "baseline"), None))
    ]

    word = "less" if sum(gaps) < 0 else "more"
    ratio = round(sum(ratios) / len(ratios)) if ratios else None
    points = f"{_about([abs(g) for g in gaps])} points {word} accurate"
    title = f"{jev} is about {ratio}× cheaper than {llm}, and {points}." if ratio else f"{jev} is {points} than {llm}."
    lead = ""
    if cascades:
        saving = [100 * (1 - c["cost_per_1k"] / h["llm_cost_per_1k"]) for h, c in cascades]
        cgap = [100 * (c["accuracy"] - h["llm_accuracy"]) for h, c in cascades]
        routed = [100 * c["routed_share"] for _, c in cascades]
        lead = (
            f"Let {jev} answer first and hand only its unsure cases to {llm}: you stay within "
            f"{max(abs(g) for g in cgap):.1f} points of {llm} and pay {_about(saving)}% less."
        )

    findings = [
        {
            "id": "accuracy",
            "icon": "target",
            "kicker": "Accuracy",
            "title": f"{_about([abs(g) for g in gaps])} points behind {llm}",
            "text": "The gap is real: "
            + (
                f"significant in all {len(hs)} {unit}."
                if sig == len(hs)
                else f"significant in {sig} of {len(hs)} {unit}."
            )
            + (
                f" Against {cheap_name} it is only {_span([-100 * p['difference'] for _, p in cheap])} points, "
                + (
                    f"not significant in any {unit[:-1]}."
                    if not any(p["p_value"] < 0.05 for _, p in cheap)
                    else f"significant in {sum(1 for _, p in cheap if p['p_value'] < 0.05)} of {len(cheap)}."
                )
                if cheap
                else ""
            ),
        }
    ]
    if ratios:
        findings.append(
            {
                "id": "cost",
                "icon": "coins",
                "kicker": "Cost",
                "title": f"{ratio}× cheaper" + (f", {_about(speeds)}× faster" if speeds else ""),
                "text": f"{_money(hs[0]['jev_cost_per_1k'])} per 1,000 calls against {_money(hs[0]['llm_cost_per_1k'])} for {llm}. "
                f"At a million calls a month: ${1000 * hs[0]['jev_cost_per_1k']:,.0f} against ${1000 * hs[0]['llm_cost_per_1k']:,.0f}.",
            }
        )
    if cascades:
        findings.append(
            {
                "id": "cascade",
                "icon": "split",
                "kicker": "Cascade",
                "title": f"{llm} accuracy for {_about(saving)}% less",
                "text": f"{jev} keeps the {_about([100 - x for x in routed])}% of items it is sure about and passes the rest to {llm}. "
                f"Accuracy lands {_span([abs(g) for g in cgap])} points below {llm} alone.",
            }
        )
    if lang_deltas:
        k = sum(1 for d in lang_deltas if d["p_value"] < 0.05)
        names = ", ".join(d["name"] for d in lang_deltas)
        findings.append(
            {
                "id": "languages",
                "icon": "languages",
                "kicker": "Languages",
                "title": "No significant drop outside " + LANGUAGE_NAMES.get(langs["reference"], langs["reference"])
                if k == 0
                else f"Drops in {k} of {len(lang_deltas)} languages",
                "text": f"{jev} loses {_span([-100 * d['difference'] for d in lang_deltas])} points in {names}; "
                f"{k} of {len(lang_deltas)} significant on the same {arm(picks[0][1], jev)['n']} items.",
            }
        )

    envelopes = {str(i): _envelope(r["business"]["arms"]) for i, r in picks}
    colors = {}
    for _, r in picks:
        for a in r["arms"]:
            colors[a["name"]] = {"jev": "var(--c-jev)", "cascade": "var(--c-cascade)", "baseline": "var(--c-base)"}.get(
                a["family"], "var(--c-llm)" if a["name"] == llm else "var(--c-llm2)"
            )
    env0 = envelopes[str(picks[0][0])]
    seg = {s["name"]: s for s in env0}
    if cascade_name in seg:
        c = seg[cascade_name]
        verdict_title = f"Use {jev} as the front door, not as the whole house"
        where = f"In {tabs[0]['name']}, " if by_lang else ""
        verdict_lede = (
            (
                f"{where}{jev} alone only wins when a mistake costs under {_price(seg[jev]['to'])}. "
                if jev in seg and seg[jev]["to"]
                else ""
            )
            + ("from" if where and not (jev in seg and seg[jev]["to"]) else "From")
            + f" {_price(c['from'])} to {_price(c['to'])} per mistake the cascade is the cheapest option all-in"
            + (f"; above that, {llm} on its own pays for itself." if c["to"] else ".")
            if c["to"]
            else f"From {_price(c['from'])} per mistake upwards the cascade is the cheapest option all-in."
        )
    else:
        verdict_title = "Pick by the price of a mistake"
        verdict_lede = "Each option below is the cheapest all-in for a range of prices of one mistake."
    n = arm(picks[0][1], jev)["n"]
    ci = picks[0][1]["arms"][0]["ci"]
    caveats = [
        f"{n} items per {unit[:-1]}: the 95% interval on one accuracy is about ±{50 * (ci[1] - ci[0]):.0f} points, so "
        "small differences need the paired tests in the deep dive.",
        "Cost covers API calls only; a model that runs locally shows $0 because hosting is not measured.",
    ]
    if baseline_gaps and max(baseline_gaps) > -5:
        caveats.append(
            f"A classic model trained on labelled data lands {_span(baseline_gaps, '{:+.1f}')} points from {jev}. "
            f"{jev} and the LLMs worked from label names only, so with labelled data a classic model is a serious option."
        )
    if any(r["dataset"].startswith(("massive", "clinc", "banking")) for _, r in picks):
        caveats.append("Public benchmark data: it may be in model training data and is not client traffic.")

    return {
        "jev": jev,
        "llm": llm,
        "cheap": cheap_name,
        "cascade": cascade_name,
        "unit": unit,
        "tabs": tabs,
        "title": title,
        "lead": lead,
        "findings": findings,
        "stats": {
            "ratio": ratio,
            "gap": sum(gaps) / len(gaps),
            "saving": sum(saving) / len(saving) if cascades else None,
            "lang_drop": -100 * sum(d["difference"] for d in lang_deltas) / len(lang_deltas) if lang_deltas else None,
            "lang_sig": sum(1 for d in lang_deltas if d["p_value"] < 0.05) if lang_deltas else None,
        },
        "lang_deltas": lang_deltas,
        "envelopes": envelopes,
        "labels": {a["name"]: _story_label(a["name"]) for _, r in picks for a in r["business"]["arms"]},
        "caveats": caveats,
        "colors": colors,
        "verdict_title": verdict_title,
        "verdict_lede": verdict_lede,
        "reference_name": LANGUAGE_NAMES.get(langs["reference"], langs["reference"]) if langs else None,
    }


ICONS = {
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "coins": '<circle cx="8" cy="8" r="6"/><path d="M18.09 10.37A6 6 0 1 1 10.34 18"/><path d="M7 6h1v4"/><path d="m16.71 13.88.7.71-2.82 2.82"/>',
    "split": '<path d="M16 3h5v5"/><path d="M8 3H3v5"/><path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/><path d="m15 9 6-6"/>',
    "languages": '<path d="m5 8 6 6"/><path d="m4 14 6-6 2-3"/><path d="M2 5h12"/><path d="M7 2h1"/><path d="m22 22-5-10-5 10"/><path d="M14 18h6"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "down": '<path d="m6 9 6 6 6-6"/>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
}


def _icon(name: str, cls: str = "icon") -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def _hero(story: dict, meta: str) -> str:
    s = story["stats"]
    stats = [
        (f'<span data-count="{abs(s["gap"]):.1f}" data-dec="1">0</span>', "points", f"behind {story['llm']}"),
    ]
    if s["ratio"]:
        stats.insert(0, (f'<span class="spot" data-count="{s["ratio"]}">0</span>', "×", f"cheaper than {story['llm']}"))
    if s["saving"] is not None:
        stats.append((f'<span data-count="{s["saving"]:.0f}">0</span>', "%", "saved by the cascade"))
    if s["lang_drop"] is not None:
        stats.append(
            (
                f'<span data-count="{s["lang_drop"]:.1f}" data-dec="1">0</span>',
                "points",
                f"average drop outside {story['reference_name']}"
                + (" (not significant)" if s["lang_sig"] == 0 else ""),
            )
        )
    tiles = "".join(
        f'<div class="hero-stat"><div class="hero-num">{num}<small>{unit}</small></div><div class="hero-note">{_esc(note)}</div></div>'
        for num, unit, note in stats
    )
    return f"""
<header class="hero" id="top">
  <div class="hero-bg" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span><span class="blob b3"></span><span class="grid-bg"></span></div>
  <div class="hero-inner">
    <p class="eyebrow reveal">{_esc(meta)}</p>
    <h1 class="reveal">{_esc(story["title"])}</h1>
    <p class="hero-lead reveal">{_esc(story["lead"])}</p>
    <div class="hero-stats reveal">{tiles}</div>
    <a class="scroll-cue" href="#summary" aria-label="Read the findings">{_icon("down")}</a>
  </div>
</header>"""


def _findings(story: dict) -> str:
    cards = "".join(
        f'<a class="finding reveal" href="#{f["id"]}" style="--d:{i * 90}ms"><span class="finding-n">0{i + 1}</span>'
        f'<span class="finding-kicker">{_icon(f["icon"])}{_esc(f["kicker"])}</span>'
        f"<strong>{_esc(f['title'])}</strong><span>{_esc(f['text'])}</span></a>"
        for i, f in enumerate(story["findings"])
    )
    return f"""
<section class="band" id="summary">
  <div class="wrap">
    <p class="kicker reveal">Summary</p>
    <h2 class="reveal">What we found</h2>
    <div class="findings">{cards}</div>
  </div>
</section>"""


def _lang_switch(story: dict) -> str:
    if len(story["tabs"]) < 2:
        return ""
    pills = "".join(
        f'<button type="button" class="pill{" on" if j == 0 else ""}" data-tab="{j}" title="{_esc(t["name"])}">{_esc(t["code"])}</button>'
        for j, t in enumerate(story["tabs"])
    )
    return f'<div class="switch" role="group" aria-label="Show results for">{pills}</div>'


def _chapter(cid: str, n: int, icon: str, kicker: str, title: str, text: str, body: str) -> str:
    return f"""
<section class="chapter" id="{cid}">
  <div class="wrap chapter-grid">
    <div class="chapter-text reveal">
      <p class="kicker">{_icon(icon)} {n:02d} · {_esc(kicker)}</p>
      <h2>{_esc(title)}</h2>
      {text}
    </div>
    <div class="chapter-viz reveal">{body}</div>
  </div>
</section>"""


def _story_sections(story: dict, runs: list[dict]) -> str:
    f = {x["id"]: x for x in story["findings"]}
    jev, llm = _esc(story["jev"]), _esc(story["llm"])
    parts = [
        _chapter(
            "accuracy",
            1,
            "target",
            "Accuracy",
            f["accuracy"]["title"],
            f"<p>{_esc(f['accuracy']['text'])}</p><p class='muted'>Bars are accuracy on the same items; the thin line is "
            "the 95% interval. Switch language at the top.</p>",
            "<div class='viz-card'><div class='viz-head'><span>Accuracy</span><span class='muted mono' data-bind='tabname'></span></div>"
            "<div class='bars' id='acc-bars'></div></div>",
        )
    ]
    if "cost" in f:
        parts.append(
            _chapter(
                "cost",
                2,
                "coins",
                "Cost and speed",
                f["cost"]["title"],
                f"<p>{_esc(f['cost']['text'])}</p><p>Accuracy is close; the price is not. Same arms, same items, same order as "
                "the chart above.</p>",
                "<div class='viz-card'><div class='viz-head'><span>$ per 1,000 calls</span><span class='muted mono' data-bind='tabname'></span></div>"
                "<div class='bars' id='cost-bars'></div></div>"
                "<div class='viz-card'><div class='viz-head'><span>Median seconds per answer</span></div><div class='bars' id='speed-bars'></div></div>",
            )
        )
    if "cascade" in f:
        parts.append(
            _chapter(
                "cascade",
                len(parts) + 1,
                "split",
                "Cascade",
                f["cascade"]["title"],
                f"<p>{_esc(f['cascade']['text'])}</p><p>{jev} returns a real probability with every answer. Above a threshold "
                f"picked on a separate calibration split, {jev} answers; below it, {llm} does.</p>"
                "<div class='legend'><span><i class='sw' style='background:var(--c-jev)'></i>answered by "
                f"{jev}</span><span><i class='sw' style='background:var(--c-llm)'></i>sent to {llm}</span></div>",
                "<div class='viz-card'><div class='viz-head'><span>100 items, routed</span><span class='muted mono' data-bind='tabname'></span></div>"
                "<div class='dots' id='cascade-dots'></div></div>"
                "<div class='viz-split'><div class='viz-card'><div class='viz-head'><span>Accuracy</span></div><div class='bars compact' id='casc-acc'></div></div>"
                "<div class='viz-card'><div class='viz-head'><span>$ per 1,000 calls</span></div><div class='bars compact' id='casc-cost'></div></div></div>",
            )
        )
    if "languages" in f:
        chips = "".join(
            f"<li><span class='mono'>{_esc(d['lang'].upper())}</span> <strong>{_pts(d['difference'])}</strong> "
            f"<span class='muted'>{_p(d['p_value'])} · {'significant' if d['p_value'] < 0.05 else 'not significant'}</span></li>"
            for d in story["lang_deltas"]
        )
        parts.append(
            _chapter(
                "languages",
                len(parts) + 1,
                "languages",
                "Languages",
                f["languages"]["title"],
                f"<p>{_esc(f['languages']['text'])}</p><p class='muted'>Every language uses the same items, translated by people, "
                f"so a change is down to the language. {jev} against English:</p><ul class='chips'>{chips}</ul>",
                "<div class='viz-card'><div class='viz-head'><span>Accuracy by language</span></div><div id='slope'></div></div>",
            )
        )
    parts.append(_decide(story, len(parts) + 1))
    parts.append(_verdict(story, runs))
    return "".join(parts)


def _decide(story: dict, n: int) -> str:
    return f"""
<section class="chapter decide" id="decide">
  <div class="wrap">
    <div class="reveal">
      <p class="kicker">{_icon("scale")} {n:02d} · Decide</p>
      <h2>The right pick depends on what a mistake costs you</h2>
      <p class="lede">Every option trades API cost against wrong answers. Put a price on one mistake (a manual fix, a refund, a lost
      customer) and the cheapest option all-in follows. Ten seconds of someone's time at $40 an hour is about $0.11.</p>
    </div>
    <div class="viz-card reveal">
      <div class="controls">
        <label><span>Cost of one mistake</span><output id="cpe-out" class="mono">$0.50</output>
          <input id="cpe" type="range" min="-3" max="2" step="0.01" value="-0.30103" aria-label="Cost of one mistake, log scale"></label>
        <label><span>Calls per month</span><output id="vol-out" class="mono">100,000</output>
          <input id="vol" type="range" min="3" max="7" step="0.01" value="5" aria-label="Calls per month, log scale"></label>
      </div>
      <div id="strip" class="strip"></div>
      <p class="answer" id="answer" aria-live="polite"></p>
      <div class="bars stacked" id="allin"></div>
      <div class="legend small"><span><i class="sw solid"></i>API calls</span><span><i class="sw hatch"></i>cost of mistakes</span></div>
    </div>
  </div>
</section>"""


def _verdict(story: dict, runs: list[dict]) -> str:
    panels = ""
    for j, t in enumerate(story["tabs"]):
        env = story["envelopes"][str(t["index"])]
        cards = ""
        for k, seg in enumerate(env):
            if seg["to"] is None:
                when = f"over {_price(seg['from'])}"
            elif seg["from"] == 0:
                when = f"under {_price(seg['to'])}"
            else:
                when = f"{_price(seg['from'])} to {_price(seg['to'])}"
            cards += (
                f"<div class='rule' style='--c:{story['colors'].get(seg['name'], 'var(--c-base)')};--d:{k * 80}ms'><span class='rule-when'>"
                f"A mistake costs {_esc(when)}</span><strong>{_esc(story['labels'].get(seg['name'], seg['name']))}</strong></div>"
            )
        panels += f"<div class='rules{' on' if j == 0 else ''}' data-tabpanel='{j}'>{cards}</div>"
    caveats = "".join(f"<li>{_icon('alert')}<span>{_esc(c)}</span></li>" for c in story["caveats"])
    return f"""
<section class="band verdict-band" id="verdict">
  <div class="wrap">
    <p class="kicker reveal">{_icon("check")} Verdict</p>
    <h2 class="reveal">{_esc(story["verdict_title"])}</h2>
    <p class="lede reveal">{_esc(story["verdict_lede"])} Cheapest option by price of one mistake
    (<span data-bind="tabname">{_esc(story["tabs"][0]["name"])}</span>):</p>
    <div class="reveal">{panels}</div>
    <h3 class="reveal">Read with care</h3>
    <ul class="caveats reveal">{caveats}</ul>
  </div>
</section>"""


def _deep_dive(runs: list[dict], langs: dict | None) -> str:
    tabs = "".join(
        f'<button type="button" class="tab{" on" if i == 0 else ""}" data-run="{i}" role="tab">'
        f"{_esc((r['language'] or '').upper() or r['id'])}<small>{_esc(r['dataset'])}</small></button>"
        for i, r in enumerate(runs)
    )
    return f"""
<section class="deep" id="deep">
  <div class="wrap">
    <p class="kicker">{_icon("layers")} Deep dive</p>
    <h2>Every number behind the story</h2>
    <p class="lede">Tables, significance tests, calibration, per-class scores and the most confident mistakes, per run.</p>
    {_summary_section(runs)}
    {_languages_section(langs)}
    {_comparison_section(runs)}
    <div class="tabs" role="tablist">{tabs}</div>
    {"".join(_run_section(r, i) for i, r in enumerate(runs))}
  </div>
</section>"""


CSS = """
:root{color-scheme:light;
--dr-green:#38b580;--dr-indigo:#445ba7;--dr-coral:#f37d82;--dr-ink:#212121;--dr-midnight:#0f172a;
--page:#f6f7f4;--surface:#ffffff;--surface-2:#eef1ec;--text:#1a1d21;--text-2:#4b5058;--muted:#7c828a;
--grid:#e4e7e2;--axis:#c6cbc4;--border:rgba(15,23,42,.09);--shadow:0 1px 2px rgba(15,23,42,.04),0 8px 28px rgba(15,23,42,.06);
--c-jev:#2e9d6c;--c-llm:#3d52a0;--c-llm2:#b05fc8;--c-cascade:#0f9fb4;--c-base:#9b9a94;
--seq-100:#d8f0e4;--seq-400:#38b580;--seq-700:#145238;--best:#e3f4ea}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0b1120;--surface:#111a2e;--surface-2:#16213a;--text:#eef1f6;--text-2:#b8c0cc;--muted:#8590a0;
--grid:#1f2a42;--axis:#2c3a57;--border:rgba(255,255,255,.08);--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.25);
--c-jev:#2fa673;--c-llm:#6a86ee;--c-llm2:#c060b8;--c-cascade:#0aa89a;--c-base:#7d7c76;
--seq-100:#12301f;--seq-400:#2fa673;--seq-700:#c9f0da;--best:#14352a}}
:root[data-theme="dark"]{color-scheme:dark;
--page:#0b1120;--surface:#111a2e;--surface-2:#16213a;--text:#eef1f6;--text-2:#b8c0cc;--muted:#8590a0;
--grid:#1f2a42;--axis:#2c3a57;--border:rgba(255,255,255,.08);--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.25);
--c-jev:#2fa673;--c-llm:#6a86ee;--c-llm2:#c060b8;--c-cascade:#0aa89a;--c-base:#7d7c76;
--seq-100:#12301f;--seq-400:#2fa673;--seq-700:#c9f0da;--best:#14352a}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:72px}
body{margin:0;background:var(--page);color:var(--text);font:15px/1.6 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;overflow-x:hidden}
.mono,code,td.num,.hero-num,.finding-n{font-family:"Roboto Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
.wrap{max-width:1120px;margin:0 auto;padding:0 20px}
h1,h2,h3{font-weight:700;letter-spacing:-.02em}
h2{font-size:clamp(26px,3.4vw,38px);line-height:1.12;margin:6px 0 16px}
h3{font-size:17px;margin:40px 0 6px}
p{margin:0 0 14px}
.muted{color:var(--muted)}
.icon{width:1em;height:1em;vertical-align:-.14em;flex:none}
.kicker{display:flex;align-items:center;gap:8px;margin:0;font:600 12.5px/1 "Roboto Mono",monospace;text-transform:uppercase;letter-spacing:.08em;color:var(--dr-green)}
.lede{font-size:17px;color:var(--text-2);max-width:68ch}
a{color:var(--c-llm)}

/* top bar */
.topbar{position:fixed;inset:0 0 auto 0;z-index:50;height:56px;display:flex;align-items:center;gap:16px;padding:0 20px;
 background:color-mix(in srgb,var(--page) 78%,transparent);backdrop-filter:blur(14px) saturate(1.4);-webkit-backdrop-filter:blur(14px) saturate(1.4);
 border-bottom:1px solid transparent;transition:border-color .3s,background .3s,transform .4s;transform:translateY(-100%)}
.topbar.show{transform:none;border-bottom-color:var(--border)}
.brand{font-weight:700;letter-spacing:-.01em;white-space:nowrap;color:var(--text);text-decoration:none}
.brand b{color:var(--dr-green)}
.nav{display:flex;gap:2px;overflow-x:auto;scrollbar-width:none;flex:1;min-width:0}
.nav a{color:var(--text-2);text-decoration:none;font-size:13.5px;padding:6px 10px;border-radius:8px;white-space:nowrap;transition:color .2s,background .2s}
.nav a:hover{color:var(--text)}.nav a.on{color:var(--text);background:var(--surface-2)}
.progress{position:absolute;left:0;bottom:-1px;height:2px;width:100%;transform-origin:0 50%;transform:scaleX(0);background:linear-gradient(90deg,var(--dr-green),var(--dr-indigo))}
.switch{display:inline-flex;padding:3px;border-radius:999px;background:var(--surface-2);border:1px solid var(--border);gap:2px;flex:none}
.pill{font:600 12px/1 "Roboto Mono",monospace;border:0;background:transparent;color:var(--text-2);padding:7px 10px;border-radius:999px;cursor:pointer;transition:all .25s}
.pill.on{background:var(--surface);color:var(--text);box-shadow:0 1px 3px rgba(0,0,0,.12)}
.iconbtn{border:1px solid var(--border);background:var(--surface);color:var(--text-2);width:34px;height:34px;border-radius:10px;display:grid;place-items:center;cursor:pointer;flex:none}

/* hero */
.hero{position:relative;min-height:100vh;display:flex;align-items:center;overflow:hidden;background:var(--dr-midnight);color:#eef2f8;isolation:isolate}
.hero-bg{position:absolute;inset:0;z-index:-1}
.blob{position:absolute;border-radius:50%;filter:blur(70px);opacity:.55;animation:drift 18s ease-in-out infinite alternate}
.b1{width:46vw;height:46vw;left:-10vw;top:-12vw;background:#38b580}
.b2{width:40vw;height:40vw;right:-8vw;top:10vh;background:#445ba7;animation-delay:-6s}
.b3{width:30vw;height:30vw;left:30vw;bottom:-14vw;background:#1f8a9e;opacity:.35;animation-delay:-11s}
.grid-bg{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.05) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.05) 1px,transparent 1px);
 background-size:56px 56px;mask-image:radial-gradient(ellipse at 50% 40%,#000 30%,transparent 75%);-webkit-mask-image:radial-gradient(ellipse at 50% 40%,#000 30%,transparent 75%)}
@keyframes drift{to{transform:translate(6vw,4vh) scale(1.12)}}
.hero-inner{max-width:1120px;margin:0 auto;padding:110px 20px 90px;width:100%}
.eyebrow{font:500 13px/1.5 "Roboto Mono",monospace;color:#9fb0c8;letter-spacing:.02em}
.hero h1{font-size:clamp(34px,5.6vw,68px);line-height:1.04;margin:14px 0 22px;max-width:17ch;letter-spacing:-.035em}
.hero-lead{font-size:clamp(17px,1.7vw,21px);color:#c7d2e2;max-width:58ch;line-height:1.5}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;margin-top:44px;border-radius:18px;overflow:hidden;
 background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.10)}
.hero-stat{padding:22px 22px 20px;background:rgba(15,23,42,.72);backdrop-filter:blur(6px)}
.hero-num{font-size:clamp(34px,4vw,48px);font-weight:600;line-height:1;letter-spacing:-.03em;color:#fff}
.hero-num small{font-size:.42em;color:#9fb0c8;margin-left:4px;letter-spacing:0}
.hero-num .spot{color:var(--dr-coral)}
.hero-note{margin-top:8px;font-size:13.5px;color:#9fb0c8}
.scroll-cue{position:absolute;left:50%;bottom:26px;transform:translateX(-50%);color:#9fb0c8;width:40px;height:40px;display:grid;place-items:center;
 border:1px solid rgba(255,255,255,.18);border-radius:50%;animation:bob 2.2s ease-in-out infinite}
.scroll-cue .icon{width:20px;height:20px}
@keyframes bob{50%{transform:translate(-50%,7px)}}

/* sections */
.band{padding:110px 0}
.chapter{padding:96px 0;border-top:1px solid var(--grid)}
.chapter-grid{display:grid;grid-template-columns:minmax(0,5fr) minmax(0,7fr);gap:56px;align-items:start}
.chapter-text{position:sticky;top:96px}
.chapter-text p{color:var(--text-2);font-size:16px}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:34px}
.finding{position:relative;display:flex;flex-direction:column;gap:10px;padding:24px 22px 26px;border-radius:18px;background:var(--surface);border:1px solid var(--border);
 box-shadow:var(--shadow);color:var(--text);text-decoration:none;overflow:hidden;transition:transform .35s cubic-bezier(.2,.8,.2,1),box-shadow .35s;transition-delay:var(--d,0ms)}
.finding:hover{transform:translateY(-4px);box-shadow:0 2px 4px rgba(15,23,42,.06),0 18px 40px rgba(15,23,42,.12)}
.finding::after{content:"";position:absolute;inset:auto 0 0 0;height:3px;background:linear-gradient(90deg,var(--dr-green),var(--dr-indigo));transform:scaleX(0);transform-origin:0;transition:transform .5s}
.finding:hover::after{transform:scaleX(1)}
.finding-n{position:absolute;right:18px;top:16px;font-size:13px;color:var(--muted)}
.finding-kicker{display:flex;gap:8px;align-items:center;font-size:12.5px;font-weight:600;color:var(--dr-green);text-transform:uppercase;letter-spacing:.06em}
.finding strong{font-size:21px;line-height:1.2;letter-spacing:-.015em}
.finding span:last-child{font-size:14px;color:var(--text-2)}

/* viz */
.viz-card{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:22px 22px 18px;box-shadow:var(--shadow);margin-bottom:16px}
.viz-head{display:flex;justify-content:space-between;gap:12px;font-size:13px;font-weight:600;color:var(--text-2);margin-bottom:14px}
.viz-split{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.viz-split .viz-card{margin-bottom:0}
.bars{display:flex;flex-direction:column;gap:12px}
.bar-row{display:grid;grid-template-columns:minmax(90px,150px) 1fr 76px;align-items:center;gap:12px;font-size:13.5px}
.bars.stacked .bar-row{grid-template-columns:minmax(120px,190px) 1fr 76px}
.bars.compact .bar-row{grid-template-columns:minmax(70px,104px) 1fr 62px;font-size:12.5px}
.bar-label{color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bar-value{text-align:right;font:500 13px "Roboto Mono",monospace;color:var(--text);white-space:nowrap}
.bar-track{position:relative;height:26px;border-radius:6px;background:var(--surface-2)}
.bars.compact .bar-track{height:18px}
.bar-fill{position:absolute;inset:0 auto 0 0;width:0;border-radius:6px;background:var(--c);transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
.bar-seg{position:absolute;top:0;bottom:0;border-radius:6px;transition:left 1s cubic-bezier(.2,.8,.2,1),width 1s cubic-bezier(.2,.8,.2,1)}
.bar-seg.hatch{background:repeating-linear-gradient(135deg,var(--c) 0 3px,transparent 3px 7px);box-shadow:inset 0 0 0 1.5px var(--c)}
.bar-ci{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--text);opacity:0;transition:opacity .6s .9s,left 1s,width 1s}
.bar-ci::before,.bar-ci::after{content:"";position:absolute;top:-4px;width:2px;height:10px;background:var(--text)}
.bar-ci::before{left:0}.bar-ci::after{right:0}
.in .bar-ci{opacity:.55}
.bar-row.best .bar-label{font-weight:700}
.bar-row[data-tip]{cursor:default}
.dots{display:grid;grid-template-columns:repeat(20,1fr);gap:6px}
.dot{aspect-ratio:1;border-radius:50%;background:var(--surface-2);transform:scale(.3);transition:background .5s,transform .6s cubic-bezier(.3,1.6,.5,1)}
.in .dot{transform:scale(1)}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--text-2);margin:10px 0 0}
.legend.small{font-size:12.5px;margin-top:14px}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.sw.solid{background:var(--text-2)}.sw.hatch{background:repeating-linear-gradient(135deg,var(--text-2) 0 2px,transparent 2px 5px);box-shadow:inset 0 0 0 1px var(--text-2)}
.chips{list-style:none;padding:0;margin:8px 0 0;display:flex;flex-direction:column;gap:8px}
.chips li{display:flex;gap:10px;align-items:baseline;padding:10px 14px;border-radius:12px;background:var(--surface);border:1px solid var(--border);font-size:14px}
#slope svg{width:100%;height:auto;display:block;overflow:visible}
.slope-line{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round;transition:stroke-dashoffset 1.8s cubic-bezier(.4,0,.2,1),opacity .3s}
.slope-dot{transition:opacity .4s 1.2s,r .2s;opacity:0}
.in .slope-dot{opacity:1}
.slope-dot:hover{r:7}
.slope-col{fill:var(--surface-2);transition:x .5s cubic-bezier(.2,.8,.2,1)}

/* decide */
.decide .viz-card{padding:28px;margin-top:28px}
.controls{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-bottom:26px}
.controls label{display:grid;grid-template-columns:1fr auto;gap:8px 12px;align-items:baseline;font-size:13.5px;color:var(--text-2);font-weight:600}
.controls output{font-size:22px;color:var(--text);font-weight:600}
.controls input{grid-column:1/-1;width:100%;accent-color:var(--dr-green)}
input[type=range]{-webkit-appearance:none;appearance:none;height:6px;border-radius:99px;background:linear-gradient(90deg,var(--dr-green),var(--dr-indigo));outline-offset:6px}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;border-radius:50%;background:var(--surface);border:3px solid var(--dr-green);box-shadow:0 2px 8px rgba(0,0,0,.2);cursor:grab}
input[type=range]::-moz-range-thumb{width:18px;height:18px;border-radius:50%;background:var(--surface);border:3px solid var(--dr-green);cursor:grab}
.strip svg{width:100%;height:auto;display:block;overflow:visible}
.strip text{font:500 11px "Roboto Mono",monospace;fill:var(--muted)}
.strip .seg-label{font:600 12px Inter,sans-serif;fill:#fff}
.strip rect.seg{transition:opacity .3s}
.strip .marker{transition:transform .15s ease-out}
.answer{font-size:clamp(18px,2vw,22px);line-height:1.4;margin:22px 0 18px;letter-spacing:-.01em}
.answer b{color:var(--dr-green)}

/* verdict */
.verdict-band{background:linear-gradient(180deg,var(--surface-2),var(--page))}
.rules{display:none;gap:12px;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));margin:26px 0 8px}
.rules.on{display:grid}
.rule{position:relative;padding:20px 20px 22px 24px;border-radius:16px;background:var(--surface);border:1px solid var(--border);box-shadow:var(--shadow);overflow:hidden;
 animation:rise .6s cubic-bezier(.2,.8,.2,1) both;animation-delay:var(--d)}
.rule::before{content:"";position:absolute;left:0;top:0;bottom:0;width:5px;background:var(--c)}
.rule-when{display:block;font:500 12.5px "Roboto Mono",monospace;color:var(--muted);margin-bottom:6px}
.rule strong{font-size:18px;letter-spacing:-.01em}
@keyframes rise{from{opacity:0;transform:translateY(14px)}}
.caveats{list-style:none;padding:0;margin:0;display:grid;gap:10px;max-width:80ch}
.caveats li{display:flex;gap:12px;color:var(--text-2);font-size:14.5px}
.caveats .icon{color:var(--muted);margin-top:4px}

/* deep dive */
.deep{padding:96px 0 120px;border-top:1px solid var(--grid)}
.deep .kicker{color:var(--c-llm)}
.deep h2 .sub,.run-title .sub{font-size:14px;font-weight:400;color:var(--text-2);letter-spacing:0}
.deep-block{margin-top:40px}
.deep-block h3{font-size:20px;margin-top:0}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:56px 0 8px;position:sticky;top:64px;z-index:5;padding:8px 0;background:var(--page)}
.tab{display:flex;flex-direction:column;align-items:flex-start;gap:2px;font:600 15px Inter,sans-serif;padding:10px 16px;border-radius:12px;border:1px solid var(--border);
 background:var(--surface);color:var(--text-2);cursor:pointer;transition:all .2s}
.tab small{font:400 11.5px "Roboto Mono",monospace;color:var(--muted)}
.tab.on{color:var(--text);border-color:var(--dr-green);box-shadow:0 0 0 3px color-mix(in srgb,var(--dr-green) 18%,transparent)}
.run-panel{display:none;animation:rise .5s both}
.run-panel.on{display:block}
.run-title{font-size:20px;margin-top:20px}
.lead{margin:0 0 12px;color:var(--text-2);font-size:14px;max-width:76ch}
.verdict{font-size:16px;max-width:72ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:16px 18px}
.tile-label{font-size:13px;color:var(--text-2)}
.tile-value{font:600 28px "Roboto Mono",monospace;margin:2px 0;letter-spacing:-.02em}
.tile-note{font-size:12.5px;color:var(--muted)}
.table-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:14px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-weight:600;color:var(--text-2);white-space:nowrap;background:var(--surface-2)}
tr:last-child td{border-bottom:0}
tbody tr{transition:background .15s}tbody tr:hover{background:var(--surface-2)}
td:first-child{white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;font-size:12.5px}
.ci{color:var(--muted);font-size:11.5px}
tr.ref td{color:var(--text-2)}
.key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:baseline}
.key.jev{background:var(--c-jev)}.key.llm{background:var(--c-llm)}.key.cascade{background:var(--c-cascade)}.key.baseline{background:var(--c-base)}
.chart{background:var(--surface);border:1px solid var(--border);border-radius:14px;height:340px;overflow:hidden}
.chart.tall{height:460px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px;margin-bottom:12px}
.grid-small{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.grid-small .chart{height:240px}
.grid2>*,.grid-small>*{min-width:0;overflow:hidden}
details{margin-top:16px;border:1px solid var(--border);border-radius:14px;background:var(--surface);padding:4px 16px}
details[open]{padding-bottom:14px}
summary{cursor:pointer;font-weight:600;padding:10px 0}
details p{color:var(--text-2);font-size:14px}
code{font-size:12px;background:var(--surface-2);padding:1px 5px;border-radius:4px}
.takeaways{margin:0 0 16px;padding-left:20px;max-width:80ch;color:var(--text-2)}.takeaways li{margin:4px 0}
.calc{display:flex;gap:16px;flex-wrap:wrap;margin:0 0 12px}.calc label{display:flex;flex-direction:column;gap:4px;font-size:13px;color:var(--text-2)}
.calc input{font:inherit;padding:7px 10px;width:180px;background:var(--surface);color:var(--text);border:1px solid var(--axis);border-radius:8px}
tr.best td{font-weight:600;background:var(--best)}
footer{padding:36px 20px 60px;text-align:center;color:var(--muted);font:12.5px/1.6 "Roboto Mono",monospace;border-top:1px solid var(--grid)}

/* tooltip and reveal */
.tip{position:fixed;z-index:100;pointer-events:none;padding:8px 11px;border-radius:10px;background:var(--surface);color:var(--text);border:1px solid var(--axis);
 box-shadow:var(--shadow);font-size:12.5px;line-height:1.45;opacity:0;transform:translateY(4px);transition:opacity .15s,transform .15s;max-width:260px}
.tip.on{opacity:1;transform:none}
.reveal{opacity:0;transform:translateY(26px);transition:opacity .9s cubic-bezier(.2,.8,.2,1),transform .9s cubic-bezier(.2,.8,.2,1);transition-delay:var(--d,0ms)}
.reveal.in{opacity:1;transform:none}
@media (max-width:860px){
 .chapter-grid{grid-template-columns:1fr;gap:28px}.chapter-text{position:static}
 .controls{grid-template-columns:1fr}.viz-split{grid-template-columns:1fr}
 .nav{display:none}.band,.chapter{padding:72px 0}
 .bar-row{grid-template-columns:minmax(72px,110px) 1fr 64px;gap:8px}
 .dots{gap:4px}
}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}.reveal{opacity:1;transform:none}}
"""

JS = r"""
const R = JSON.parse(document.getElementById('report-data').textContent);
const S = R.story;
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const $ = (s, el = document) => el.querySelector(s), $$ = (s, el = document) => [...el.querySelectorAll(s)];
function esc(x){return String(x).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
function pct(x){return (100*x).toFixed(1)+'%';}
function money(x){return x==null?'–':x>=100?'$'+Math.round(x).toLocaleString('en-US'):x>=1?'$'+x.toFixed(2):x===0?'$0':'$'+x.toFixed(x<0.01?4:3);}

// ---- reveal on scroll, count-up numbers, top bar ---------------------------------------------
const io = new IntersectionObserver(es => es.forEach(e => {
  if (!e.isIntersecting) return;
  e.target.classList.add('in'); io.unobserve(e.target);
  $$('[data-count]', e.target).forEach(countUp);
}), {threshold: .18, rootMargin: '0px 0px -40px 0px'});
function countUp(el){
  const to = +el.dataset.count, dec = +(el.dataset.dec || 0), t0 = performance.now(), dur = reduce ? 0 : 1600;
  const step = now => { const k = Math.min(1, (now - t0) / (dur || 1)), e = 1 - Math.pow(1 - k, 4);
    el.textContent = (to * e).toFixed(dec); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
$$('.reveal, .chapter-viz, .viz-card').forEach(el => io.observe(el));
if (reduce) $$('.reveal').forEach(el => el.classList.add('in'));

const bar = $('.topbar'), prog = $('.progress');
const navLinks = $$('.nav a'), targets = navLinks.map(a => document.getElementById(a.hash.slice(1))).filter(Boolean);
function onScroll(){
  const y = scrollY, h = document.documentElement.scrollHeight - innerHeight;
  bar.classList.toggle('show', y > innerHeight * 0.55);
  prog.style.transform = `scaleX(${h > 0 ? y / h : 0})`;
  let cur = null; for (const t of targets) if (t.getBoundingClientRect().top < 140) cur = t.id;
  navLinks.forEach(a => a.classList.toggle('on', a.hash === '#' + cur));
}
addEventListener('scroll', onScroll, {passive: true}); onScroll();

// ---- tooltip -----------------------------------------------------------------------------------
const tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip);
document.addEventListener('pointermove', e => {
  const t = e.target.closest('[data-tip]');
  if (!t) { tip.classList.remove('on'); return; }
  tip.innerHTML = t.dataset.tip; tip.classList.add('on');
  const w = tip.offsetWidth, x = Math.min(innerWidth - w - 12, e.clientX + 14);
  tip.style.left = x + 'px'; tip.style.top = (e.clientY + 16) + 'px';
});

// ---- theme -------------------------------------------------------------------------------------
const themeBtn = $('#theme');
try { const saved = localStorage.getItem('jev-theme'); if (saved) document.documentElement.dataset.theme = saved; } catch (e) {}
themeBtn?.addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('jev-theme', document.documentElement.dataset.theme); } catch (e) {}
});

// ---- story charts ------------------------------------------------------------------------------
function colorOf(a){ return S.colors[a.name] || 'var(--c-base)'; }
// Fixed order so colour and position follow the arm, whatever the language.
function storyArms(run){
  const by = n => run.arms.find(a => a.name === n);
  const llms = run.arms.filter(a => a.family === 'llm' && a.name !== S.llm).sort((a, b) => a.cost_per_1k - b.cost_per_1k);
  return [by(S.jev), ...llms, by(S.llm), ...run.arms.filter(a => a.family === 'baseline')].filter(Boolean);
}
// Keyed bars: an update animates widths in place instead of redrawing.
function bars(id, rows, max){
  const el = document.getElementById(id); if (!el) return;
  rows.forEach((r, j) => {
    let row = el.querySelector(`[data-key="${CSS.escape(r.key)}"]`);
    if (!row) {
      row = document.createElement('div'); row.className = 'bar-row'; row.dataset.key = r.key;
      row.innerHTML = `<div class="bar-label"></div><div class="bar-track"></div><div class="bar-value"></div>`;
      el.appendChild(row);
    }
    row.style.order = j;
    row.classList.toggle('best', !!r.best);
    row.dataset.tip = r.tip;
    row.querySelector('.bar-label').textContent = r.label;
    row.querySelector('.bar-value').textContent = r.text;
    const track = row.querySelector('.bar-track');
    const segs = r.segs || [{v: r.value, c: r.color}];
    while (track.querySelectorAll('.bar-seg').length < segs.length) { const s = document.createElement('div'); s.className = 'bar-seg'; s.style.left = '0'; s.style.width = '0'; track.appendChild(s); }
    let left = 0;
    track.querySelectorAll('.bar-seg').forEach((s, k) => {
      const seg = segs[k]; if (!seg) { s.style.width = '0'; return; }
      const w = max > 0 ? 100 * seg.v / max : 0;
      s.className = 'bar-seg' + (seg.hatch ? ' hatch' : ''); s.style.setProperty('--c', seg.c);
      s.style.background = seg.hatch ? '' : seg.c;
      // A 2px surface gap between stacked segments; a non-zero value never shrinks below 3px.
      const l = left, gap = k ? 2 : 0; left += w;
      const apply = () => { s.style.left = `calc(${l}% + ${gap}px)`; s.style.width = `max(${seg.v > 0 ? 3 : 0}px, calc(${w}% - ${gap}px))`; };
      if (el.closest('.in') || reduce) requestAnimationFrame(apply); else s._pending = apply;
    });
    let ci = track.querySelector('.bar-ci');
    if (r.ci) {
      if (!ci) { ci = document.createElement('div'); ci.className = 'bar-ci'; track.appendChild(ci); }
      ci.style.left = (100 * r.ci[0] / max) + '%'; ci.style.width = (100 * (r.ci[1] - r.ci[0]) / max) + '%';
    }
  });
  [...el.children].forEach(c => { if (!rows.some(r => r.key === c.dataset.key)) c.remove(); });
}
// Bars drawn before their card is visible grow when it scrolls in.
const growIo = new IntersectionObserver(es => es.forEach(e => { if (!e.isIntersecting) return;
  setTimeout(() => $$('.bar-seg', e.target).forEach(s => { s._pending?.(); s._pending = null; }), 150); growIo.unobserve(e.target); }), {threshold: .25});
$$('.bars').forEach(b => growIo.observe(b));

let tab = 0;
function run(){ return R.runs[S.tabs[tab].index]; }

function drawAccuracy(){
  const arms = storyArms(run()), top = Math.max(...arms.map(a => a.accuracy));
  bars('acc-bars', arms.map(a => ({key: a.name, label: a.label, value: a.accuracy, color: colorOf(a), ci: a.ci, text: pct(a.accuracy), best: a.accuracy === top,
    tip: `<b>${esc(a.label)}</b><br>Accuracy ${pct(a.accuracy)}<br>95% interval ${pct(a.ci[0])} to ${pct(a.ci[1])}<br>${a.n} items`})), 1);
}
function drawCost(){
  const arms = storyArms(run()), max = Math.max(...arms.map(a => a.cost_per_1k || 0));
  bars('cost-bars', arms.map(a => ({key: a.name, label: a.label, value: a.cost_per_1k || 0, color: colorOf(a), text: a.cost_per_1k ? money(a.cost_per_1k) : 'local',
    tip: `<b>${esc(a.label)}</b><br>${money(a.cost_per_1k)} per 1,000 calls<br>${money(1000 * a.cost_per_1k)} per million`})), max);
  const sp = arms.filter(a => a.p50), smax = Math.max(...sp.map(a => a.p50));
  bars('speed-bars', sp.map(a => ({key: a.name, label: a.label, value: a.p50, color: colorOf(a), text: a.p50 < 0.001 ? '<1 ms' : a.p50 < 1 ? Math.round(1000 * a.p50) + ' ms' : a.p50.toFixed(2) + ' s',
    tip: `<b>${esc(a.label)}</b><br>median ${a.p50.toFixed(3)} s<br>p95 ${a.p95.toFixed(3)} s`})), smax);
}
function drawCascade(){
  const r = run(), c = Object.values(r.cascades || {}).find(c => c.fallback === S.llm); if (!c) return;
  const p = c.operating_points.match_fallback?.eval; if (!p) return;
  const box = $('#cascade-dots');
  if (!box.children.length) for (let k = 0; k < 100; k++) { const d = document.createElement('i'); d.className = 'dot'; d.style.transitionDelay = `${k * 9}ms`; box.appendChild(d); }
  const routed = Math.round(100 * p.routed_share);
  $$('.dot', box).forEach((d, k) => { d.style.background = k >= 100 - routed ? 'var(--c-llm)' : 'var(--c-jev)';
    d.dataset.tip = k >= 100 - routed ? `Sent to ${esc(S.llm)}: ${routed} of 100` : `Answered by ${esc(S.jev)}: ${100 - routed} of 100`; });
  const jev = r.arms.find(a => a.name === S.jev), llm = r.arms.find(a => a.name === S.llm);
  const rows = [{key: 'j', label: S.jev + ' alone', a: jev.accuracy, cost: jev.cost_per_1k, c: 'var(--c-jev)'},
                {key: 'c', label: 'cascade', a: p.accuracy, cost: p.cost_per_1k, c: 'var(--c-cascade)', best: true},
                {key: 'l', label: S.llm + ' alone', a: llm.accuracy, cost: llm.cost_per_1k, c: 'var(--c-llm)'}];
  bars('casc-acc', rows.map(x => ({key: x.key, label: x.label, value: x.a, color: x.c, text: pct(x.a), best: x.best, tip: `<b>${esc(x.label)}</b><br>Accuracy ${pct(x.a)}`})), 1);
  const cm = Math.max(...rows.map(x => x.cost));
  bars('casc-cost', rows.map(x => ({key: x.key, label: x.label, value: x.cost, color: x.c, text: money(x.cost), best: x.best, tip: `<b>${esc(x.label)}</b><br>${money(x.cost)} per 1,000 calls`})), cm);
}
function drawSlope(){
  const el = $('#slope'), L = R.languages; if (!el || !L) return;
  const names = [S.jev, ...Object.keys(L.arms).filter(n => n !== S.jev && n !== S.llm && L.arms[n][L.languages[0]]?.family === 'llm'), S.llm,
                 ...Object.keys(L.arms).filter(n => Object.values(L.arms[n])[0].family === 'baseline')].filter(n => L.arms[n]);
  const W = 640, H = 330, pl = 46, pr = 118, pt = 16, pb = 34, langs = L.languages;
  const vals = names.flatMap(n => langs.map(l => L.arms[n][l]?.accuracy).filter(v => v != null));
  const lo = Math.floor((Math.min(...vals) * 100 - 3) / 5) * 5, hi = Math.ceil((Math.max(...vals) * 100 + 2) / 5) * 5;
  const x = j => pl + (W - pl - pr) * (langs.length > 1 ? j / (langs.length - 1) : .5), y = v => pt + (H - pt - pb) * (1 - (100 * v - lo) / (hi - lo));
  const sel = langs.indexOf(run().language);
  let g = `<rect class="slope-col" x="${x(Math.max(0, sel)) - 26}" y="${pt - 6}" width="52" height="${H - pt - pb + 12}" rx="10"/>`;
  for (let t = lo; t <= hi; t += 5) g += `<line x1="${pl}" x2="${W - pr}" y1="${y(t / 100)}" y2="${y(t / 100)}" stroke="var(--grid)"/><text x="${pl - 10}" y="${y(t / 100) + 4}" text-anchor="end" font-size="11" fill="var(--muted)" font-family="Roboto Mono">${t}%</text>`;
  langs.forEach((l, j) => g += `<text x="${x(j)}" y="${H - 8}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text-2)" font-family="Roboto Mono">${l.toUpperCase()}</text>`);
  // End labels, nudged apart so they never overlap.
  const ends = names.map(n => ({n, v: y(L.arms[n][langs[langs.length - 1]]?.accuracy ?? vals[0])})).sort((a, b) => a.v - b.v);
  for (let k = 1; k < ends.length; k++) if (ends[k].v - ends[k - 1].v < 15) ends[k].v = ends[k - 1].v + 15;
  names.forEach(n => {
    const col = colorOf({name: n}), pts = langs.map((l, j) => [j, L.arms[n][l]]).filter(p => p[1]);
    const d = pts.map(([j, c], k) => `${k ? 'L' : 'M'}${x(j)},${y(c.accuracy)}`).join('');
    g += `<path class="slope-line" d="${d}" stroke="${col}" pathLength="1" stroke-dasharray="1" stroke-dashoffset="${el.closest('.in') ? 0 : 1}"/>`;
    pts.forEach(([j, c]) => g += `<circle class="slope-dot" cx="${x(j)}" cy="${y(c.accuracy)}" r="5" fill="${col}" stroke="var(--surface)" stroke-width="2"
      data-tip="<b>${esc(n)}</b> · ${langs[j].toUpperCase()}<br>Accuracy ${pct(c.accuracy)}${c.delta ? `<br>${(100 * c.delta.difference).toFixed(1)} pts vs ${L.reference.toUpperCase()}, p=${c.delta.p_value.toFixed(3)}` : ''}"/>`);
    const e = ends.find(e => e.n === n);
    g += `<text x="${W - pr + 12}" y="${e.v + 4}" font-size="12.5" fill="var(--text)" font-family="Inter">${esc(n)}</text>`;
  });
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Accuracy per arm and language">${g}</svg>`;
}
new IntersectionObserver((es, o) => es.forEach(e => { if (!e.isIntersecting) return;
  setTimeout(() => $$('.slope-line', e.target).forEach(p => p.style.strokeDashoffset = 0), 120); o.unobserve(e.target); }), {threshold: .3})
  .observe($('#slope') || document.body);

// ---- decide ------------------------------------------------------------------------------------
const cpeIn = $('#cpe'), volIn = $('#vol');
function decideArms(r){
  const seen = new Set();
  return r.business.arms.slice().sort((a, b) => a.name.startsWith('cascade-') - b.name.startsWith('cascade-')).filter(a => {
    const k = a.cost_per_1k.toFixed(6) + '|' + a.accuracy.toFixed(6); if (seen.has(k)) return false; seen.add(k); return true; });
}
function drawStrip(env, cpe){
  const W = 640, H = 64, lo = -3, hi = 2, x = v => W * (Math.log10(Math.max(v, 1e-3)) - lo) / (hi - lo);
  let g = '';
  env.forEach(s => {
    const x0 = x(Math.max(s.from, 1e-3)), x1 = s.to == null ? W : Math.min(W, x(s.to)); if (x1 <= 0 || x1 - x0 < .5) return;
    const col = colorOf(s);
    g += `<rect class="seg" x="${x0}" y="0" width="${x1 - x0}" height="34" fill="${col}" data-tip="<b>${esc(S.labels[s.name] || s.name)}</b><br>cheapest when a mistake costs ${money(s.from)}${s.to == null ? ' or more' : ' to ' + money(s.to)}"/>`;
    const lab = s.name.startsWith('cascade-') ? 'cascade' : (S.labels[s.name] || s.name); if (x1 - x0 > lab.length * 6.6 + 12) g += `<text class="seg-label" x="${(x0 + x1) / 2}" y="21" text-anchor="middle" pointer-events="none">${esc(lab)}</text>`;
  });
  [0.001, 0.01, 0.1, 1, 10, 100].forEach(v => g += `<line x1="${x(v)}" x2="${x(v)}" y1="36" y2="42" stroke="var(--axis)"/><text x="${x(v)}" y="56" text-anchor="${v === 0.001 ? 'start' : v === 100 ? 'end' : 'middle'}">$${v}</text>`);
  g += `<g class="marker" style="transform:translateX(${x(cpe)}px)"><line x1="0" x2="0" y1="-6" y2="40" stroke="var(--text)" stroke-width="2"/><circle cx="0" cy="-8" r="5" fill="var(--text)"/></g>`;
  $('#strip').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cheapest option by cost of one mistake, log scale">${g}</svg>`;
}
function drawDecide(){
  if (!cpeIn) return;
  const raw = Math.pow(10, +cpeIn.value), cpe = +raw.toPrecision(raw < 1 ? 1 : 2), vol = Math.round(Math.pow(10, +volIn.value) / 1000) * 1000 || 1000;
  const price = x => '$' + (x >= 0.1 ? x.toFixed(2) : String(x));
  $('#cpe-out').textContent = price(cpe); $('#vol-out').textContent = vol.toLocaleString('en-US');
  const r = run(), env = S.envelopes[String(S.tabs[tab].index)];
  drawStrip(env, cpe);
  const rows = decideArms(r).map(a => { const api = vol * a.cost_per_1k / 1000, err = vol * (1 - a.accuracy) * cpe;
    return {a, api, err, total: api + err}; }).sort((p, q) => p.total - q.total);
  const max = Math.max(...rows.map(x => x.total)), win = rows[0], next = rows[1];
  bars('allin', rows.map((x, j) => { const col = colorOf(x.a);
    return {key: x.a.name, label: S.labels[x.a.name] || x.a.name, best: j === 0, text: money(x.total),
      segs: [{v: x.api, c: col}, {v: x.err, c: col, hatch: true}],
      tip: `<b>${esc(S.labels[x.a.name] || x.a.name)}</b><br>API ${money(x.api)} / month<br>${Math.round(vol * (1 - x.a.accuracy)).toLocaleString('en-US')} mistakes: ${money(x.err)}<br>All-in ${money(x.total)}`}; }), max);
  $('#answer').innerHTML = `At <b class="mono">${price(cpe)}</b> per mistake and ${vol.toLocaleString('en-US')} calls a month, the cheapest option is
    <b>${esc(S.labels[win.a.name] || win.a.name)}</b> at ${money(win.total)} a month all-in${next ? `, ${money(next.total - win.total)} less than ${esc(S.labels[next.a.name] || next.a.name)}` : ''}.`;
}
cpeIn?.addEventListener('input', drawDecide); volIn?.addEventListener('input', drawDecide);

function setTab(j){
  tab = j;
  $$('.pill').forEach(p => p.classList.toggle('on', +p.dataset.tab === j));
  $$('[data-tabpanel]').forEach(p => p.classList.toggle('on', +p.dataset.tabpanel === j));
  $$('[data-bind="tabname"]').forEach(el => el.textContent = S.tabs[j].name);
  drawAccuracy(); drawCost(); drawCascade(); drawSlope(); drawDecide();
}
$$('.pill').forEach(p => p.addEventListener('click', () => setTab(+p.dataset.tab)));
if (S) setTab(0);

// ---- deep dive (Plotly, drawn when a run's tab is first opened) --------------------------------
const SYMBOLS = ['circle','diamond','square','triangle-up','cross','x'];
function tok(){const s=getComputedStyle(document.documentElement);return k=>s.getPropertyValue(k).trim();}
function famColor(t,f){return {jev:t('--c-jev'),llm:t('--c-llm'),cascade:t('--c-cascade'),baseline:t('--c-base')}[f]||t('--muted');}
function symbolsFor(arms){const seen={},out={};for(const a of arms){seen[a.family]=(seen[a.family]||0);out[a.name]=SYMBOLS[seen[a.family]%SYMBOLS.length];seen[a.family]++;}return out;}
function layout(t,extra){return Object.assign({paper_bgcolor:t('--surface'),plot_bgcolor:t('--surface'),
 font:{family:'Inter,system-ui,sans-serif',size:12,color:t('--text-2')},
 margin:{l:64,r:24,t:24,b:52},showlegend:false,hoverlabel:{bgcolor:t('--surface'),bordercolor:t('--axis'),font:{color:t('--text')}}},extra);}
function ax(t,extra){return Object.assign({gridcolor:t('--grid'),linecolor:t('--axis'),zeroline:false,tickfont:{color:t('--muted')},title:{font:{color:t('--text-2')}}},extra);}
const CFG={displayModeBar:false,responsive:true};

function drawRun(run,i,t){
  const rid='r'+i, sym=symbolsFor(run.arms);
  const paid=run.arms.filter(a=>a.cost_per_1k>0);
  Plotly.react(rid+'-scatter',paid.map(a=>({type:'scatter',mode:'markers+text',x:[a.cost_per_1k],y:[100*a.accuracy],
    text:[a.label],textposition:'top center',textfont:{color:t('--text'),size:12},
    error_y:{type:'data',symmetric:false,array:[100*(a.ci[1]-a.accuracy)],arrayminus:[100*(a.accuracy-a.ci[0])],color:famColor(t,a.family),thickness:1.5,width:0},
    marker:{size:12,color:famColor(t,a.family),symbol:sym[a.name],line:{color:t('--surface'),width:2}},
    hovertemplate:`<b>${a.label}</b><br>Accuracy %{y:.1f}%<br>$%{x:.4f} per 1k<extra></extra>`})),
    layout(t,{xaxis:ax(t,{type:'log',dtick:1,tickprefix:'$',title:{text:'$ per 1,000 predictions (log scale)'}}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
  const lat=run.arms.filter(a=>a.p50);
  Plotly.react(rid+'-latency',lat.map(a=>({type:'scatter',mode:'lines+markers',x:[a.p50,a.p95],y:[a.label,a.label],
    line:{color:famColor(t,a.family),width:2},marker:{size:[11,7],color:famColor(t,a.family),symbol:sym[a.name],line:{color:t('--surface'),width:2}},
    hovertemplate:`<b>${a.label}</b><br>p50 ${(a.p50).toFixed(3)} s<br>p95 ${(a.p95).toFixed(3)} s<extra></extra>`})),
    layout(t,{margin:{l:230,r:24,t:16,b:52},xaxis:ax(t,{type:'log',dtick:1,ticksuffix:' s',title:{text:'Seconds per prediction (log scale)'}}),yaxis:ax(t,{autorange:'reversed',gridcolor:t('--grid')})}),CFG);
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
       line:{color:famColor(t,a.family),width:2},marker:{size:a.bins.map(b=>6+12*Math.sqrt(b.n/n)),color:famColor(t,a.family),line:{color:t('--surface'),width:2}},
       customdata:a.bins.map(b=>b.n),hovertemplate:'Said %{x:.2f}, right %{y:.2f}<br>%{customdata} predictions<extra></extra>'}],
      layout(t,{width:d.clientWidth,height:d.clientHeight,margin:{l:44,r:12,t:34,b:38},title:{text:`${a.name} · ECE ${a.ece==null?'–':a.ece.toFixed(3)}`,font:{size:13,color:t('--text')},x:0.04},
        xaxis:ax(t,{range:[0,1.02],dtick:0.25}),yaxis:ax(t,{range:[0,1.02],dtick:0.25})}),CFG);
  });
  for(const [name,c] of Object.entries(run.cascades||{})){
    const s=c.sweep.eval, x=s.map(p=>100*p.routed_share);
    const pts=Object.entries(c.operating_points).map(([k,p])=>({k,x:100*p.eval.routed_share,acc:100*p.eval.accuracy,cost:p.eval.cost_per_1k}));
    const mk=(y,yp,title,suffix,fmt)=>[
      {type:'scatter',mode:'lines',x,y,line:{color:t('--c-cascade'),width:2},customdata:s.map(p=>p.threshold),
       hovertemplate:`Threshold %{customdata:.2f}<br>Routed %{x:.1f}%<br>${title} %{y:${fmt}}${suffix}<extra></extra>`},
      {type:'scatter',mode:'markers+text',x:pts.map(p=>p.x),y:yp,text:pts.map(p=>p.k.replace(/_/g,' ')),textposition:pts.map((p,j)=>j%2?'bottom right':'top right'),
       textfont:{color:t('--text'),size:12},marker:{size:10,color:t('--c-cascade'),line:{color:t('--surface'),width:2}},hoverinfo:'skip'}];
    Plotly.react(`${rid}-cascade-acc-${name}`,mk(s.map(p=>100*p.accuracy),pts.map(p=>p.acc),'Accuracy','%','.1f'),
      layout(t,{xaxis:ax(t,{title:{text:'Share routed to the LLM (%)'},ticksuffix:'%'}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
    Plotly.react(`${rid}-cascade-cost-${name}`,mk(s.map(p=>p.cost_per_1k),pts.map(p=>p.cost),'Cost $','','.4f'),
      layout(t,{xaxis:ax(t,{title:{text:'Share routed to the LLM (%)'},ticksuffix:'%'}),yaxis:ax(t,{title:{text:'$ per 1,000 predictions'},tickprefix:'$'})}),CFG);
  }
  const arms=run.arms.filter(a=>!a.name.startsWith('cascade-'));
  const classes=Object.keys(arms[0]?.per_class||{});
  const mean=c=>arms.reduce((s,a)=>s+(a.per_class[c]??0),0)/arms.length;
  const worst=classes.sort((a,b)=>mean(a)-mean(b)).slice(0,25);
  Plotly.react(rid+'-heatmap',[{type:'heatmap',x:arms.map(a=>a.label),y:worst,z:worst.map(c=>arms.map(a=>a.per_class[c]??null)),
    zmin:0,zmax:1,colorscale:[[0,t('--seq-100')],[0.5,t('--seq-400')],[1,t('--seq-700')]],xgap:2,ygap:2,
    colorbar:{title:{text:'F1',font:{color:t('--text-2')}},tickfont:{color:t('--muted')},outlinewidth:0,thickness:10},
    hovertemplate:'%{y}<br>%{x}: F1 %{z:.2f}<extra></extra>'}],
    layout(t,{margin:{l:220,r:24,t:16,b:90},xaxis:ax(t,{gridcolor:'rgba(0,0,0,0)'}),yaxis:ax(t,{autorange:'reversed',gridcolor:'rgba(0,0,0,0)',tickfont:{size:11,color:t('--text-2')}})}),CFG);
}

function drawCompare(t){
  if(R.runs.length<2||!document.getElementById('compare-chart')) return;
  const names=[...new Set(R.runs.flatMap(r=>r.arms.map(a=>a.name)))];
  const fam={}; R.runs.forEach(r=>r.arms.forEach(a=>fam[a.name]=a.family));
  const sym=symbolsFor(names.map(n=>({name:n,family:fam[n]})));
  Plotly.react('compare-chart',names.map(n=>({type:'scatter',mode:'lines+markers',name:n,
    x:R.runs.map(r=>r.id),y:R.runs.map(r=>{const a=r.arms.find(a=>a.name===n);return a?100*a.accuracy:null;}),
    line:{color:famColor(t,fam[n]),width:2},marker:{size:9,color:famColor(t,fam[n]),symbol:sym[n],line:{color:t('--surface'),width:2}},
    hovertemplate:`<b>${n}</b><br>%{x}<br>Accuracy %{y:.1f}%<extra></extra>`})),
    layout(t,{showlegend:true,legend:{orientation:'h',y:-0.25,font:{color:t('--text-2')}},margin:{l:64,r:24,t:16,b:120},
      xaxis:ax(t,{}),yaxis:ax(t,{title:{text:'Accuracy (%)'},ticksuffix:'%'})}),CFG);
}

function initCalc(run,i){
  const rid='r'+i, vol=document.getElementById(rid+'-vol'), cpe=document.getElementById(rid+'-cpe'), body=document.getElementById(rid+'-calc');
  if(!body) return;
  const usd=x=>'$'+(Math.abs(x)>=100?Math.round(x).toLocaleString('en-US'):x.toFixed(2));
  function draw(){
    const v=+vol.value||0, c=+cpe.value||0;
    const rows=run.business.arms.map(a=>{const api=v*a.cost_per_1k/1000, err=v*(1-a.accuracy);return {name:a.name,api,err,errCost:err*c,total:api+err*c};}).sort((x,y)=>x.total-y.total);
    body.innerHTML=rows.map((r,j)=>`<tr${j===0?' class="best"':''}><td>${esc(r.name)}</td><td class="num">${usd(r.api)}</td><td class="num">${Math.round(r.err).toLocaleString('en-US')}</td><td class="num">${usd(r.errCost)}</td><td class="num">${usd(r.total)}</td></tr>`).join('');
  }
  vol.addEventListener('input',draw); cpe.addEventListener('input',draw); draw();
}

const drawn = new Set();
let deepSeen = false;
function showRun(i){
  $$('.tab').forEach(b => b.classList.toggle('on', +b.dataset.run === i));
  $$('.run-panel').forEach(p => p.classList.toggle('on', +p.dataset.run === i));
  if (deepSeen && window.Plotly && !drawn.has(i)) { drawRun(R.runs[i], i, tok()); drawn.add(i); }
}
$$('.tab').forEach(b => b.addEventListener('click', () => showRun(+b.dataset.run)));
R.runs.forEach((r, i) => initCalc(r, i));
function redrawDeep(){ if (!window.Plotly || !deepSeen) return; const t = tok(); drawn.forEach(i => drawRun(R.runs[i], i, t)); drawCompare(t); }
document.addEventListener('DOMContentLoaded', () => {
  showRun(0);
  // Plotly is heavy: draw the deep dive only once the reader gets near it.
  const deep = document.getElementById('deep');
  new IntersectionObserver((es, o) => { if (!es[0].isIntersecting || !window.Plotly) return; o.disconnect(); deepSeen = true;
    const t = tok(); const on = +($('.tab.on')?.dataset.run || 0); drawRun(R.runs[on], on, t); drawn.add(on); drawCompare(t); }, {rootMargin: '600px'}).observe(deep);
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawDeep);
new MutationObserver(redrawDeep).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
"""


def build(run_dirs: list[Path], out: Path, title: str = "Jev vs LLMs") -> Path:
    runs = [_run_payload(d) for d in run_dirs]
    languages = _languages(runs)
    story = _story(runs, languages)
    versions = sorted({v for r in runs for a in r["arms"] if a["family"] == "jev" for v in a["versions"]})
    spend = sum(r["manifest"].get("new_spend_usd", 0) for r in runs)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    footer = (
        f"Jev version {', '.join(versions) or '–'} · generated {generated} · API spend for these runs {_money(spend)}"
    )
    datasets = sorted({r["dataset"].split("_")[0] for r in runs})
    n = runs[0]["arms"][0]["n"] if runs and runs[0]["arms"] else 0
    meta = f"{title} · {', '.join(datasets)} · {n} items" + (
        f" × {len(languages['languages'])} languages"
        if languages
        else (f" × {len(runs)} runs" if len(runs) > 1 else "")
    )
    nav = [
        ("summary", "Summary"),
        ("accuracy", "Accuracy"),
        ("cost", "Cost"),
        ("cascade", "Cascade"),
        ("languages", "Languages"),
        ("decide", "Decide"),
        ("verdict", "Verdict"),
        ("deep", "Deep dive"),
    ]
    if story:
        have = {f["id"] for f in story["findings"]} | {"summary", "decide", "verdict", "deep"}
        nav = [(k, v) for k, v in nav if k in have]
        top = _hero(story, meta) + _findings(story) + _story_sections(story, runs)
    else:
        nav = [("deep", "Results")]
        top = f"""<header class="hero" id="top"><div class="hero-bg" aria-hidden="true"><span class="blob b1"></span><span class="blob b2"></span>
<span class="grid-bg"></span></div><div class="hero-inner"><p class="eyebrow">{_esc(meta)}</p><h1>{_esc(title)}</h1>
<p class="hero-lead">Quality, cost, latency and calibration on the same data and inputs. These runs have no Jev and LLM pair to compare, so
the results below are shown without a summary.</p></div></header>"""
    links = "".join(f'<a href="#{k}">{_esc(v)}</a>' for k, v in nav)
    data = json.dumps({"runs": runs, "languages": languages, "story": story}).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<script src="{PLOTLY}" defer></script></head>
<body>
<div class="topbar"><a class="brand" href="#top"><b>jev</b> vs llms</a><nav class="nav">{links}</nav>
{_lang_switch(story) if story else ""}<button type="button" class="iconbtn" id="theme" aria-label="Toggle dark mode">{_icon("moon")}</button>
<span class="progress"></span></div>
{top}
<main>
{_deep_dive(runs, languages)}
</main>
<footer>dataroots · {_esc(footer)}</footer>
<script type="application/json" id="report-data">{data}</script>
<script>{JS}</script>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
