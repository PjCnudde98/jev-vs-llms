"""Arm registry: maps the `kind` in a run config to its class."""

from jev_eval.arms.base import Arm, RetryableError
from jev_eval.arms.baseline import BaselineArm
from jev_eval.arms.jev import JevArm
from jev_eval.arms.llm import AnthropicArm, OpenAIArm

ARMS: dict[str, type[Arm]] = {cls.kind: cls for cls in (JevArm, AnthropicArm, OpenAIArm, BaselineArm)}

__all__ = ["ARMS", "Arm", "RetryableError"]
