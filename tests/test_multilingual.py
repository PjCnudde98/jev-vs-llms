"""MASSIVE preparation, the Japanese-capable baseline, and the paired comparison. All offline."""

import asyncio
import io
import json
import tarfile

import pytest

from jev_eval import datasets, metrics
from jev_eval.arms.baseline import BaselineArm
from jev_eval.types import FieldSpec, Item, TaskSpec, load_items

INTENTS = ["alarm_set", "alarm_query", "weather_query", "general_greet"]
TEXT = {
    "en": {
        "alarm_set": "wake me at five",
        "alarm_query": "what alarms do I have",
        "weather_query": "is it raining",
        "general_greet": "hello",
    },
    "fr": {
        "alarm_set": "réveille-moi à cinq heures",
        "alarm_query": "quelles alarmes ai-je",
        "weather_query": "il pleut",
        "general_greet": "bonjour",
    },
    "nl": {
        "alarm_set": "maak me wakker om vijf uur",
        "alarm_query": "welke alarmen heb ik",
        "weather_query": "regent het",
        "general_greet": "hallo",
    },
    "ja": {
        "alarm_set": "五時に起こして",
        "alarm_query": "アラームを教えて",
        "weather_query": "雨が降っていますか",
        "general_greet": "こんにちは",
    },
}
LOCALES = {"en": "en-US", "fr": "fr-FR", "nl": "nl-NL", "ja": "ja-JP"}


def _fake_massive(path):
    """A tiny archive shaped like the real one: the same ids in every locale."""
    with tarfile.open(path, "w:gz") as tar:
        for code, locale in LOCALES.items():
            rows = []
            for n in range(40):
                intent = INTENTS[n % 4]
                # general_greet appears once per partition, like the real dev/test sets
                if intent == "general_greet" and n >= 8:
                    intent = "alarm_set"
                partition = "train" if n < 20 else "dev" if n < 30 else "test"
                rows.append(
                    {
                        "id": str(n),
                        "locale": locale,
                        "partition": partition,
                        "intent": intent,
                        "utt": TEXT[code][intent],
                    }
                )
            data = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows)).encode()
            info = tarfile.TarInfo(f"1.1/data/{locale}.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def massive_root(tmp_path, monkeypatch):
    (tmp_path / "data" / "raw" / "massive").mkdir(parents=True)
    _fake_massive(tmp_path / "data" / "raw" / "massive" / "massive-1.1.tar.gz")
    monkeypatch.setattr(datasets, "MASSIVE_EVAL_SIZE", 6)
    monkeypatch.setattr(datasets, "MASSIVE_CALIBRATION_SIZE", 4)
    datasets.prepare_massive(tmp_path)
    return tmp_path


def test_massive_ids_are_paired_across_languages(massive_root):
    def ids(code, split):
        return [i.id for i in load_items(massive_root / "data" / "processed" / f"massive_{code}" / f"{split}.jsonl")]

    for split in ("calibration", "eval"):
        assert ids("fr", split) == ids("en", split) == ids("ja", split) == ids("nl", split)
    assert len(ids("en", "eval")) == 6 and len(ids("en", "calibration")) == 4


def test_massive_text_is_translated_but_labels_and_task_stay_english(massive_root):
    ja = load_items(massive_root / "data" / "processed" / "massive_ja" / "eval.jsonl")
    en = {i.id: i for i in load_items(massive_root / "data" / "processed" / "massive_en" / "eval.jsonl")}
    assert all(i.labels == en[i.id].labels for i in ja)
    assert all(i.meta["language"] == "ja" for i in ja)
    assert any(any(ord(c) > 0x3000 for c in i.text) for i in ja)
    task = TaskSpec.load(massive_root / "tasks" / "massive_ja.yaml")
    assert "Japanese" in task.context
    assert task.fields[0].labels == sorted(INTENTS)
    assert task.fields[0].descriptions["alarm_set"] == "Alarm: set"


def test_massive_is_deterministic(massive_root):
    path = massive_root / "data" / "processed" / "massive_nl" / "eval.jsonl"
    first = path.read_bytes()
    datasets.prepare_massive(massive_root)
    assert path.read_bytes() == first


def _ja_items():
    return [
        Item(id=f"{label}-{n}", text=f"{TEXT['ja'][label]}{'です' * n}", labels={"intent": label})
        for label in INTENTS
        for n in range(6)
    ]


def test_char_ngram_baseline_learns_japanese():
    task = TaskSpec(name="ja", fields=[FieldSpec(name="intent", instruction="?", labels=INTENTS)])
    items = _ja_items()
    # The default word tokenizer sees each unspaced Japanese phrase as one token and
    # can only match whole phrases; character n-grams generalise to unseen suffixes.
    arm = BaselineArm("baseline", "baseline-tfidf-logreg", task, analyzer="char_wb")
    arm.fit(items)
    probe = Item(id="p", text="雨が降っていますかね", labels={"intent": "weather_query"})
    prediction = asyncio.run(arm.predict(probe))
    assert prediction.answers["intent"].label == "weather_query"


def _rows(correct: list[bool]):
    return [
        {"item_id": f"i{n}", "repeat": 0, "gold": {"intent": "a"}, "answers": {"intent": {"label": "a" if ok else "b"}}}
        for n, ok in enumerate(correct)
    ]


def test_paired_difference_detects_a_real_gap_and_ignores_noise():
    strong = _rows([True] * 100)
    weak = _rows([True] * 60 + [False] * 40)
    gap = metrics.paired_difference(strong, weak, "intent")
    assert gap["difference"] == pytest.approx(0.4)
    assert gap["only_a_right"] == 40 and gap["only_b_right"] == 0
    assert gap["p_value"] < 0.001 and gap["difference_ci"][0] > 0

    same = metrics.paired_difference(weak, weak, "intent")
    assert same["difference"] == 0 and same["p_value"] == 1.0


def test_paired_difference_needs_shared_items():
    other = [{**r, "item_id": f"x{n}"} for n, r in enumerate(_rows([True] * 5))]
    with pytest.raises(ValueError, match="share no items"):
        metrics.paired_difference(_rows([True] * 5), other, "intent")
