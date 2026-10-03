"""
Parallel Jev scoring shared by the live scanner and the calibration harness.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Mapping, Optional

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from engine.jev_questions import VERSIONS, QuestionSet, strategy_for

#: Published per-million input-token price the scanner has been reporting.
#: Not verified against a TypeSafe invoice; treat cost figures as estimates.
EST_USD_PER_M_INPUT = 0.042


def api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    key = os.getenv("JEV_API_KEY") or os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("JEV_API_KEY missing in .env or environment.")
    return key


async def _score_one(client, sem, item_id: str, snap: Mapping[str, Any], qset: QuestionSet, model: Optional[str]) -> Dict[str, Any]:
    strategy = strategy_for(snap)
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.system_one(state=dict(snap), questions=qset.questions[strategy], model=model)
        except Exception as exc:  # noqa: BLE001 -- one failed symbol must not sink the batch
            return {"id": item_id, "strategy": strategy, "error": f"{type(exc).__name__}: {exc}"}
        latency = (time.perf_counter() - t0) * 1000.0
    tokens = resp.usage.input_tokens or 0
    return {
        "id": item_id,
        "strategy": strategy,
        "scores": {k: round(v.noul, 4) for k, v in resp.nouls.items()},
        "latency_ms": round(latency, 1),
        "tokens": tokens,
        "model": resp.model,
    }


async def score_many(
    items: Mapping[str, Mapping[str, Any]],
    version: str,
    *,
    model: Optional[str] = None,
    concurrency: int = 8,
) -> List[Dict[str, Any]]:
    """Score {id: snapshot} concurrently. Results come back in input order."""
    qset = VERSIONS[version]
    sem = asyncio.Semaphore(concurrency)
    retry = RetryPolicy(max_retries=4, http_statuses={429, 500, 502, 503, 504})
    async with AsyncTypeSafeClient(api_key=api_key(), retry=retry) as client:
        return await asyncio.gather(*(_score_one(client, sem, i, s, qset, model) for i, s in items.items()))
