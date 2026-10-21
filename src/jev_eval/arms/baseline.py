"""Arm E: a classic floor. TF-IDF features + logistic regression, trained on the
train split. Runs locally, costs nothing, and tells us how much of the task a model
with no language understanding already solves.
"""

from __future__ import annotations

import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from jev_eval.arms.base import Arm
from jev_eval.types import FieldAnswer, Item, Prediction


class BaselineArm(Arm):
    kind = "baseline"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._models: dict[str, Pipeline] = {}

    def fit(self, train: list[Item]) -> None:
        if not train:
            raise ValueError("The baseline arm needs the train split.")
        texts = [i.text for i in train]
        for field in self.task.fields:
            model = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                LogisticRegression(max_iter=3000, C=self.params.get("C", 10.0)),
            )
            model.fit(texts, [i.labels[field.name] for i in train])
            self._models[field.name] = model

    async def predict(self, item: Item) -> Prediction:
        start = time.perf_counter()
        answers = {}
        for field in self.task.fields:
            model = self._models[field.name]
            row = model.predict_proba([item.text])[0]
            probs = {label: float(p) for label, p in zip(model.classes_, row, strict=True) if label in field.labels}
            label = max(probs, key=probs.get)
            answers[field.name] = FieldAnswer(label=label, probs=probs, confidence=probs[label])
        return Prediction(
            arm=self.name,
            model=self.model,
            model_version=self.model,
            item_id=item.id,
            answers=answers,
            gold=item.labels,
            latency_s=time.perf_counter() - start,
        )
