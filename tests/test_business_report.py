"""Cost calculator, paired significance in the report, and the by-language section. No API needed."""

import asyncio
import json

import pytest
import yaml

from jev_eval import business, cascade, config, metrics, report, runner
from jev_eval.types import Item, write_jsonl


def _arm(name, accuracy, cost_per_1k, model="m"):
    return {"name": name, "model": model, "accuracy": accuracy, "cost_per_1k": cost_per_1k, "p95": 1.0}


ARMS = [
    _arm("jev", 0.90, 0.02, "jev-1.13.0"),
    _arm("haiku", 0.92, 0.20, "claude-haiku-5-5"),
    _arm("opus", 0.96, 6.00, "claude-opus-5-5"),
    _arm("baseline", 0.70, 0.0, "baseline-tfidf-logreg"),
    _arm("slow-bad", 0.85, 1.00, "other"),
]


def test_reference_is_the_most_accurate_llm():
    assert business.default_reference(ARMS) == "opus"


def test_monthly_cost_and_mistakes():
    result = business.analyse(ARMS, volumes=[100_000], cost_per_error=2.0, reference="opus")
    jev = next(a for a in result["arms"] if a["name"] == "jev")
    cell = jev["monthly"]["100000"]
    assert cell["api_usd"] == pytest.approx(2.0)  # 100k calls x $0.02 per 1k
    assert cell["errors"] == pytest.approx(10_000)
    assert cell["total_usd"] == pytest.approx(2.0 + 20_000)


def test_break_even_cost_per_error_is_where_the_totals_cross():
    result = business.analyse(ARMS, volumes=[1000], reference="opus")
    jev = next(a for a in result["arms"] if a["name"] == "jev")
    v = jev["vs_reference"]
    assert v["verdict"] == "cheaper_but_less_accurate"
    assert v["accuracy_points"] == pytest.approx(-6.0)
    # Opus saves 6 points of mistakes for $5.98 per 1k calls: $5.98 / 60 mistakes
    assert v["break_even_cost_per_error"] == pytest.approx(5.98 / 60)
    # At exactly that price both cost the same.
    at_break_even = business.analyse(ARMS, volumes=[1000], cost_per_error=v["break_even_cost_per_error"])
    totals = {a["name"]: a["monthly"]["1000"]["total_usd"] for a in at_break_even["arms"]}
    assert totals["jev"] == pytest.approx(totals["opus"])


def test_dominated_arms_are_flagged_and_verdicts_cover_each_case():
    result = business.analyse(ARMS, volumes=[1000], reference="opus")
    by = {a["name"]: a for a in result["arms"]}
    assert by["slow-bad"]["dominated_by"] == ["haiku", "jev"]
    assert by["jev"]["dominated_by"] == []
    assert by["slow-bad"]["vs_reference"]["verdict"] == "cheaper_but_less_accurate"
    jev_ref = business.analyse(ARMS, volumes=[1000], reference="jev")
    assert next(a for a in jev_ref["arms"] if a["name"] == "opus")["vs_reference"]["verdict"] == (
        "costlier_but_more_accurate"
    )
    assert next(a for a in jev_ref["arms"] if a["name"] == "slow-bad")["vs_reference"]["verdict"] == "dominated"
    assert next(a for a in jev_ref["arms"] if a["name"] == "baseline")["vs_reference"]["verdict"] == (
        "cheaper_but_less_accurate"
    )


def test_unknown_reference_is_an_error():
    with pytest.raises(ValueError, match="Unknown reference"):
        business.analyse(ARMS, reference="nope")


def test_format_table_lists_every_arm():
    text = business.format_table(business.analyse(ARMS, cost_per_error=1.0))
    assert all(a["name"] in text for a in ARMS)
    assert "Break-even" in text and "All-in" in text


@pytest.fixture
def two_languages(tmp_path, monkeypatch, task):
    for module in (config, runner, metrics):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "tasks").mkdir()
    labels = ["billing", "shipping", "returns", "oos"]
    # Accuracy of the two models by language: Jev drops a lot in `fr`, the large model does not.
    accuracy = {"en": (0.9, 0.95), "fr": (0.55, 0.93)}
    runs = {}
    for lang, (jev_acc, big_acc) in accuracy.items():
        items = [
            Item(
                id=f"i{n:03d}",
                text=f"message {n}",
                labels={"intent": labels[n % 4]},
                meta={"language": lang},
            )
            for n in range(120)
        ]
        name = f"toy_{lang}"
        (tmp_path / "tasks" / f"{name}.yaml").write_text(yaml.safe_dump({**task.model_dump(), "name": name}))
        data = tmp_path / "data" / "processed" / name
        write_jsonl(data / "train.jsonl", items)
        write_jsonl(data / "calibration.jsonl", items[:60])
        write_jsonl(data / "eval.jsonl", items[60:])
        cfg = {
            "name": name,
            "dataset": name,
            "splits": ["calibration", "eval"],
            "max_cost_usd": 50.0,
            "arms": [
                {
                    "name": "jev",
                    "kind": "fake",
                    "model": "jev-fake",
                    "params": {"accuracy": jev_acc, "cost": 0.00002, "latency": 0.1},
                },
                {
                    "name": "opus",
                    "kind": "fake",
                    "model": "fake-large",
                    "params": {"accuracy": big_acc, "cost": 0.006, "latency": 2.0},
                },
            ],
        }
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        out = asyncio.run(runner.run(path, f"run-{lang}"))
        cascade.run(out, "jev", "opus", ["intent"])
        runs[lang] = out
    return tmp_path, runs


def _data(page):
    return json.loads(page.read_text().split('id="report-data">')[1].split("</script>")[0])


def test_report_has_summary_calculator_and_language_sections(two_languages):
    root, runs = two_languages
    page = report.build([runs["en"], runs["fr"]], root / "report.html")
    text = page.read_text()
    for heading in ("Summary", "By language", "Is the difference real?", "What it costs at scale", "Your numbers"):
        assert heading in text
    assert 'id="r0-vol"' in text and 'id="r1-calc"' in text

    data = _data(page)
    assert [r["language"] for r in data["runs"]] == ["en", "fr"]
    fr = data["runs"][1]
    assert fr["headline"]["jev"] == "jev" and fr["headline"]["llm"] == "opus"
    assert fr["headline"]["cascade"] is not None
    assert fr["headline"]["paired"]["difference"] < 0 and fr["headline"]["paired"]["p_value"] < 0.05
    assert fr["pairs"][0]["a"] == "jev" and fr["business"]["reference"] == "opus"


def test_language_deltas_are_paired_against_the_reference_language(two_languages):
    root, runs = two_languages
    data = _data(report.build([runs["fr"], runs["en"]], root / "report.html"))
    langs = data["languages"]
    assert langs["languages"] == ["en", "fr"] and langs["reference"] == "en"  # English first, whatever the order passed
    jev = langs["arms"]["jev"]
    assert jev["en"]["delta"] is None
    assert jev["fr"]["delta"]["difference"] < -0.2 and jev["fr"]["delta"]["p_value"] < 0.01
    assert abs(langs["arms"]["opus"]["fr"]["delta"]["difference"]) < 0.1


def test_single_language_has_no_language_section(two_languages):
    root, runs = two_languages
    page = report.build([runs["en"]], root / "report.html")
    assert "By language" not in page.read_text()
    assert _data(page)["languages"] is None
