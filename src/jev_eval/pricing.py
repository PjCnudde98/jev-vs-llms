"""Turn token counts into dollars, using configs/pricing.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PRICING_PATH = Path(__file__).resolve().parents[2] / "configs" / "pricing.yaml"


class MissingPriceError(KeyError):
    pass


@lru_cache
def load_prices(path: str = str(PRICING_PATH)) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text()) or {}


def price(model: str) -> dict:
    prices = load_prices()
    if model not in prices:
        raise MissingPriceError(
            f"No price for model '{model}' in {PRICING_PATH}. Add it before running, "
            "so cost numbers are never silently zero."
        )
    return prices[model]


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    p = price(model)
    per_token = 1e-6
    cache_read = p.get("cache_read", p["input"] * 0.1)
    cache_write = p.get("cache_write", p["input"] * 1.25)
    return per_token * (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_read_tokens * cache_read
        + cache_write_tokens * cache_write
    )
