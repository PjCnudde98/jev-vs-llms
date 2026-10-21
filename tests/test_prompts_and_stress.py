from jev_eval import prompts, stress
from jev_eval.config import StressConfig


def test_llm_schema_restricts_labels(task):
    schema = prompts.llm_output_schema(task)
    assert schema["properties"]["intent"]["properties"]["label"]["enum"] == task.fields[0].labels
    assert schema["required"] == ["intent"]


def test_every_arm_gets_the_same_descriptions(task, items):
    examples = prompts.select_examples(items, task, 2)
    system = prompts.llm_system_prompt(task, examples)
    criteria = prompts.jev_criteria(task.fields[0], examples["intent"])
    assert "- billing: Charges and invoices" in system
    assert criteria["billing"]["description"] == "Charges and invoices"
    assert criteria["billing"]["examples"] == examples["intent"]["billing"]
    assert len(examples["intent"]["billing"]) == 2


def test_examples_are_deterministic(task, items):
    assert prompts.select_examples(items, task, 3) == prompts.select_examples(list(reversed(items)), task, 3)


def test_max_classes_keeps_out_of_scope(task, items):
    kept_items, kept_task = stress.apply(items, task, StressConfig(max_classes=1))
    labels = kept_task.fields[0].labels
    assert len(labels) == 2 and "oos" in labels
    assert {i.labels["intent"] for i in kept_items} <= set(labels)


def test_noise_is_seeded_per_item(task, items):
    cfg = StressConfig(typos=0.2, boilerplate=True)
    a, _ = stress.apply(items, task, cfg)
    b, _ = stress.apply(items, task, cfg)
    assert [i.text for i in a] == [i.text for i in b]
    assert "Original Message" in a[0].text and a[0].text != items[0].text
